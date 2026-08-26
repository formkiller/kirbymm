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
    float *Cl = C;   /* C 初值加载用指针 (推到末尾) */
    float *Cw = C;   /* 写回用指针 (独立, 不受加载推进影响) */
    __asm__ volatile(
        "smstart sm\n\t"
        "smstart za\n\t"
        "ptrue p0.s, all\n\t"
        /* 加载 C 初值进 za0 (完整 C += A*B 语义, 代替 zero {za}) */
        "mov w12, #0\n\t"
        "3:\n\t"
        "ld1w {za0h.s[w12, 0]}, p0/z, [%[Cl]]\n\t"
        "add %[Cl], %[Cl], #64\n\t"
        "add w12, w12, #1\n\t"
        "cmp w12, #16\n\t"
        "b.ne 3b\n\t"
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
        "st1w {za0h.s[w12, 0]}, p0, [%[Cw]]\n\t"
        "add %[Cw], %[Cw], #64\n\t"            /* C 下一行 */
        "add w12, w12, #1\n\t"                 /* 行号++ */
        "cmp w12, #16\n\t"
        "b.ne 2b\n\t"
        "smstop za\n\t"
        "smstop sm\n\t"
        : [A] "+r" (A), [B] "+r" (B), [Cl] "+r" (Cl), [Cw] "+r" (Cw), [K] "+r" (K)
        :
        : "memory", "w12"
    );
}

/* 32x32 K-update microkernel (论文主例程: 4 ZA tile 2x2 块, mr1=32 nr=32 mr2=0)
 * C: 32x32 row-major 输出 (覆盖式, 假设初始 0)
 * A: col-major 32 x K (每列 32 FP32 连续)
 * B: row-major K x 32 (每行 32 FP32 连续)
 * tile 布局: za0=C[0:16,0:16] za1=C[0:16,16:32] za2=C[16:32,0:16] za3=C[16:32,16:32]
 */
__attribute__((target("sve,sme")))
static void micro_32x32_kupdate(float *C, int ldC, const float *A, const float *B, int K) {
    /* ldC: C 行距 (元素数, = N 在 MacroKernel 中)
     * 直接在原 C 上操作, 避免 pack/unpack C 的 memcpy 开销 */
    float *Cl = C;               /* C 初值加载: 低 16 行起点 */
    float *Ch = C + 16 * (size_t)ldC;  /* C 初值加载: 高 16 行起点 */
    float *Cl2 = C;              /* 写回: 低 16 行起点 */
    float *Ch2 = C + 16 * (size_t)ldC;
    const float *A1 = A + 16;    /* A 列后 16 行 */
    const float *B1 = B + 16;    /* B 行后 16 列 */
    long ldc_skip = (long)ldC * 4 - 64;  /* 行间跳距(字节): 行距*4 - 已加的64 */
    /* 当 ldC=32 (连续 32×32): ldc_skip=64, 与原固定 #64 一致 */
    __asm__ volatile(
        "smstart sm\n\t"
        "smstart za\n\t"
        "ptrue p0.s, all\n\t"
        /* 加载 C 初值进 4 tile (行距 ldC, 用 ldc_skip 跳行) */
        "mov w12, #0\n\t"
        "3:\n\t"
        "ld1w {za0h.s[w12, 0]}, p0/z, [%[Cl]]\n\t"
        "add %[Cl], %[Cl], #64\n\t"
        "ld1w {za1h.s[w12, 0]}, p0/z, [%[Cl]]\n\t"
        "add %[Cl], %[Cl], %[ls]\n\t"
        "ld1w {za2h.s[w12, 0]}, p0/z, [%[Ch]]\n\t"
        "add %[Ch], %[Ch], #64\n\t"
        "ld1w {za3h.s[w12, 0]}, p0/z, [%[Ch]]\n\t"
        "add %[Ch], %[Ch], %[ls]\n\t"
        "add w12, w12, #1\n\t"
        "cmp w12, #16\n\t"
        "b.ne 3b\n\t"
        /* K 循环: 每步 2 A 向量 + 2 B 向量, 4 fmopa (2x2 tile 块) */
        "1:\n\t"
        "ld1w z0.s, p0/z, [%[A]]\n\t"
        "ld1w z1.s, p0/z, [%[A1]]\n\t"
        "ld1w z2.s, p0/z, [%[B]]\n\t"
        "ld1w z3.s, p0/z, [%[B1]]\n\t"
        "fmopa za0.s, p0/m, p0/m, z0.s, z2.s\n\t"
        "fmopa za1.s, p0/m, p0/m, z0.s, z3.s\n\t"
        "fmopa za2.s, p0/m, p0/m, z1.s, z2.s\n\t"
        "fmopa za3.s, p0/m, p0/m, z1.s, z3.s\n\t"
        "add %[A],  %[A],  #128\n\t"
        "add %[A1], %[A1], #128\n\t"
        "add %[B],  %[B],  #128\n\t"
        "add %[B1], %[B1], #128\n\t"
        "subs %w[K], %w[K], #1\n\t"
        "b.ne 1b\n\t"
        /* 写回 (行距 ldC, 用 Cl2/Ch2 + ldc_skip) */
        "mov w12, #0\n\t"
        "2:\n\t"
        "st1w {za0h.s[w12, 0]}, p0, [%[Cl2]]\n\t"
        "add %[Cl2], %[Cl2], #64\n\t"
        "st1w {za1h.s[w12, 0]}, p0, [%[Cl2]]\n\t"
        "add %[Cl2], %[Cl2], %[ls]\n\t"
        "st1w {za2h.s[w12, 0]}, p0, [%[Ch2]]\n\t"
        "add %[Ch2], %[Ch2], #64\n\t"
        "st1w {za3h.s[w12, 0]}, p0, [%[Ch2]]\n\t"
        "add %[Ch2], %[Ch2], %[ls]\n\t"
        "add w12, w12, #1\n\t"
        "cmp w12, #16\n\t"
        "b.ne 2b\n\t"
        "smstop za\n\t"
        "smstop sm\n\t"
        : [A] "+r" (A), [A1] "+r" (A1), [B] "+r" (B), [B1] "+r" (B1),
          [Cl] "+r" (Cl), [Ch] "+r" (Ch), [Cl2] "+r" (Cl2), [Ch2] "+r" (Ch2),
          [K] "+r" (K)
        : [ls] "r" (ldc_skip)
        : "memory", "w12"
    );
}

/* KirbyMM GEMM 入口 (阶段一: MacroKernel 32×32 块遍历)
 * M=N=16: 走 16×16 单瓦片 SME
 * M,N 都是 32 倍数: 32×32 块遍历, 每块走 SME (pack A/B/C)
 * 其他 (含边界): fallback naive
 * 完整 C += A*B 语义 (microkernel 加载 C 初值进 ZA, 累加后写回)
 */
void kirby_sgemm_fp32(int M, int N, int K,
                      const float *A, const float *B, float *C) {
    /* M=N=16 单块 SME 路径 (保留作小规模验证) */
    if (M == 16 && N == 16) {
        float *A_col = (float *)malloc(sizeof(float) * 16 * (size_t)K);
        for (int k = 0; k < K; k++)
            for (int i = 0; i < 16; i++)
                A_col[k * 16 + i] = A[i * K + k];
        micro_16x16_kupdate(C, A_col, B, K);
        free(A_col);
        return;
    }

    /* 非 32 倍数 (含边界) → naive 兜底 */
    if (M % 32 != 0 || N % 32 != 0) {
        naive_sgemm_fp32(M, N, K, A, B, C);
        return;
    }

    /* MacroKernel: 32×32 块遍历 (M, N 都是 32 倍数)
     * 每块 pack A (转置 col-major) + pack B (跨 stride row-major)
     * C 直接在原矩阵操作 (行距 N), 无 pack/unpack C → 省掉 2×32 memcpy/块 */
    float *A_blk = (float *)malloc(sizeof(float) * 32 * (size_t)K);
    float *B_blk = (float *)malloc(sizeof(float) * (size_t)K * 32);

    for (int ii = 0; ii < M; ii += 32) {
        for (int jj = 0; jj < N; jj += 32) {
            /* pack A[ii:ii+32, :] → col-major 32×K
             * 分块转置: 32×32 子块 (4KB 装入 L1), 减少 A 跨行 stride K 的 cache miss
             * 子块内 A 行内连续读 (32 元素), A_blk 子块内跨行 stride 32 (L1 命中) */
            for (int kk = 0; kk < K; kk += 32) {
                int blkk = (kk + 32 <= K) ? 32 : (K - kk);
                for (int k = 0; k < blkk; k++) {
                    for (int i = 0; i < 32; i++) {
                        A_blk[(kk + k) * 32 + i] = A[(ii + i) * K + (kk + k)];
                    }
                }
            }
            /* pack B[:, jj:jj+32] → row-major K×32 (跨 stride 复制, 每行连续 32) */
            for (int k = 0; k < K; k++)
                for (int j = 0; j < 32; j++)
                    B_blk[k * 32 + j] = B[k * N + (jj + j)];
            /* microkernel: 直接在原 C 上操作 (行距 N, 无 C pack/unpack) */
            micro_32x32_kupdate(C + (size_t)ii * N + jj, N, A_blk, B_blk, K);
        }
    }
    free(A_blk); free(B_blk);
}
