/*
 * primary.c — KirbyMM 主例程 microkernel (16x16 单瓦片版本)
 *
 * 当前实现 (阶段一, M4 验证):
 *   - 16x16 单 ZA 瓦片 (za0), K 维循环累加
 *   - M=N=16 时走 SME, 其他规模 fallback 到 naive
 *   - 假设 C 初始为 0, 输出 C = A*B (覆盖式, 非累加)
 *
 * 后续扩展 (DESIGN.md 开发顺序):
 *   - 32x32 (4 瓦片, 2x2 块) — 论文主例程
 *   - C 初始值累加 (C += A*B 完整语义)
 *   - MacroKernel 分块循环, 支持任意 M/N/K
 *
 * 数据布局约定:
 *   - A (microkernel 内部): col-major 16 x K, 每列 16 FP32 连续
 *   - B (microkernel 内部): row-major K x 16, 每行 16 FP32 连续
 *   - C: 16x16 row-major
 *   wrapper 负责把外部 row-major A 转成 col-major
 *
 * 关键 SME 指令链路 (已验证, 见 tools/fmopa_sanity.c):
 *   smstart sm + smstart za + zero {za} + ptrue p0.s, all
 *   ld1w z0, p0/z, [addr]   (从内存加载 16 FP32 到向量)
 *   fmopa za0.s, p0/m, p0/m, z0.s, z1.s   (外积累加进 za0)
 *   st1w {za0h.s[w12, i]}, p0, [addr]     (za0 水平 slice i 存到内存)
 *   smstop za + smstop sm
 *
 * 注意: index 指令在 M4 streaming 模式内生成 0 (已知问题),
 *       microkernel 用 ld1w 从打包缓冲加载, 不依赖 index.
 */
#include "kirby.h"
#include <stdlib.h>
#include <string.h>

/* 16x16 K-update microkernel (单 ZA 瓦片)
 * C: 16x16 row-major 输出 (覆盖式, 假设初始 0)
 * A: col-major 16 x K (每列 16 FP32 连续)
 * B: row-major K x 16 (每行 16 FP32 连续)
 */
__attribute__((target("sve,sme")))
static void micro_16x16_kupdate(float *C, const float *A, const float *B, int K) {
    __asm__ volatile(
        "smstart sm\n\t"
        "smstart za\n\t"
        "zero {za}\n\t"
        "ptrue p0.s, all\n\t"
        /* K 循环: 每步加载 A 一列 + B 一行, fmopa 累加进 za0 */
        "1:\n\t"
        "ld1w z0.s, p0/z, [%[A]]\n\t"           /* A 列 16 FP32 */
        "ld1w z1.s, p0/z, [%[B]]\n\t"           /* B 行 16 FP32 */
        "fmopa za0.s, p0/m, p0/m, z0.s, z1.s\n\t"
        "add %[A], %[A], #64\n\t"              /* A 下一列 (16*4=64 字节) */
        "add %[B], %[B], #64\n\t"              /* B 下一行 */
        "subs %w[K], %w[K], #1\n\t"             /* K-- (w 寄存器, 32-bit) */
        "b.ne 1b\n\t"
        /* 写回: za0 的 16 行存到 C (row-major, 每行 64 字节)
         * ZA slice 语义: 行号 = w12 + imm, imm ∈ [0,3] (2-bit 编码)
         * 故用循环: w12 递增 0..15, imm 恒 0 */
        "mov w12, #0\n\t"                       /* 行号 = 0 */
        "2:\n\t"
        "st1w {za0h.s[w12, 0]}, p0, [%[C]]\n\t"
        "add %[C], %[C], #64\n\t"              /* C 下一行 */
        "add w12, w12, #1\n\t"                 /* 行号++ */
        "cmp w12, #16\n\t"
        "b.ne 2b\n\t"
        "smstop za\n\t"
        "smstop sm\n\t"
        : [A] "+r" (A), [B] "+r" (B), [C] "+r" (C), [K] "+r" (K)
        :
        : "memory", "w12"
    );
}

/* KirbyMM GEMM 入口 (当前简化版, 阶段一)
 * M=N=16 时走 SME microkernel, 其他规模 fallback naive
 * C += A*B 语义 (但当前 microkernel 覆盖式, 假设 C 初始 0)
 */
void kirby_sgemm_fp32(int M, int N, int K,
                      const float *A, const float *B, float *C) {
    if (M != 16 || N != 16) {
        /* 非 16 规模暂用 naive 兜底, 后续接 MacroKernel 分块 */
        naive_sgemm_fp32(M, N, K, A, B, C);
        return;
    }

    /* A: row-major 16xK → col-major 16xK (转置, 每列连续) */
    float *A_col = (float *)malloc(sizeof(float) * 16 * (size_t)K);
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < 16; i++) {
            A_col[k * 16 + i] = A[i * K + k];   /* A_col[k][i] = A[i][k] */
        }
    }

    /* B: row-major Kx16 已符合 microkernel 约定 (每行连续), 无需转换 */

    /* zero C (microkernel 覆盖式输出, 假设初始 0) */
    for (int i = 0; i < 16 * 16; i++) C[i] = 0.0f;

    micro_16x16_kupdate(C, A_col, B, K);

    free(A_col);
}
