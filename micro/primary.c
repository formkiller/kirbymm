/*
 * primary.c — KirbyMM 主例程 (论文 §IV-A 严格复现版)
 *
 * 严格按论文 Fig.7 实现, 不添加额外优化:
 *   - micro_32x32: 每次迭代 4 fmopa, K 循环 (不展开), C 连续 32×32
 *   - C += A*B 完整语义 (加载 C 初值进 ZA, fmopa 累加, 写回)
 *   - MacroKernel: 32×32 块遍历 + pack A/B/C (后续 §V-A/§V-B 优化)
 *
 * 论文 §IV-A 描述 (Fig.7):
 *   "In each iteration, two vector registers are used to load elements from
 *    matrix A, and another two vector registers load elements from matrix B
 *    to perform an outer product. The results are then accumulated into a
 *    2×2 block of ZA tiles."
 *
 * 数据布局约定 (microkernel 内部):
 *   A: col-major 32 × K (每列 32 FP32 连续, 供 ld1w 加载)
 *   B: row-major K × 32 (每行 32 FP32 连续, 供 ld1w 加载)
 *   C: 32×32 row-major (行距 32 = 128 字节, 连续块)
 *
 * ZA slice 语法 (M4 实测确认):
 *   行号 = w12 + imm, imm ∈ [0,3] (2-bit), w12 递增 0..15, imm 恒 0
 *   ld1w/st1w 无大立即偏移 (scaled 索引 [-8,7]), 全用独立指针 + add
 *
 * 编译: 函数级 __attribute__((target("sve,sme"))) 放行 SME/SVE 指令
 */
#include "kirby.h"
#include <stdlib.h>
#include <string.h>

/* 16×16 单瓦片 microkernel (保留作小规模验证, 非论文核心)
 * C: 16×16 row-major (行距 16 = 64 字节, 连续块)
 * A: col-major 16 × K, B: row-major K × 16
 */
__attribute__((target("sve,sme")))
static void micro_16x16_kupdate(float *C, const float *A, const float *B, int K) {
    float *Cl = C;
    float *Cw = C;
    __asm__ volatile(
        "smstart sm\n\t"
        "smstart za\n\t"
        "ptrue p0.s, all\n\t"
        /* 加载 C 初值进 za0 (完整 C += A*B 语义) */
        "mov w12, #0\n\t"
        "3:\n\t"
        "ld1w {za0h.s[w12, 0]}, p0/z, [%[Cl]]\n\t"
        "add %[Cl], %[Cl], #64\n\t"
        "add w12, w12, #1\n\t"
        "cmp w12, #16\n\t"
        "b.ne 3b\n\t"
        /* K 循环: 每步加载 A 一列 + B 一行, fmopa 累加进 za0 */
        "1:\n\t"
        "ld1w z0.s, p0/z, [%[A]]\n\t"
        "ld1w z1.s, p0/z, [%[B]]\n\t"
        "fmopa za0.s, p0/m, p0/m, z0.s, z1.s\n\t"
        "add %[A], %[A], #64\n\t"
        "add %[B], %[B], #64\n\t"
        "subs %w[K], %w[K], #1\n\t"
        "b.ne 1b\n\t"
        /* 写回: za0 的 16 行存到 C */
        "mov w12, #0\n\t"
        "2:\n\t"
        "st1w {za0h.s[w12, 0]}, p0, [%[Cw]]\n\t"
        "add %[Cw], %[Cw], #64\n\t"
        "add w12, w12, #1\n\t"
        "cmp w12, #16\n\t"
        "b.ne 2b\n\t"
        "smstop za\n\t"
        "smstop sm\n\t"
        : [A] "+r" (A), [B] "+r" (B), [Cl] "+r" (Cl), [Cw] "+r" (Cw), [K] "+r" (K)
        :
        : "memory", "w12"
    );
}

/* 32×32 主例程 microkernel (论文 §IV-A, Fig.7 严格复现)
 * C: 32×32 row-major (行距 32 = 128 字节, 连续块)
 * A: col-major 32 × K (每列 32 FP32 连续)
 * B: row-major K × 32 (每行 32 FP32 连续)
 * tile 布局: za0=C[0:16,0:16] za1=C[0:16,16:32] za2=C[16:32,0:16] za3=C[16:32,16:32]
 */
__attribute__((target("sve,sme")))
static void micro_32x32_kupdate(float *C, const float *A, const float *B, int K) {
    float *Cl = C;             /* C 初值加载: 低 16 行起点 */
    float *Ch = C + 16 * 32;   /* C 初值加载: 高 16 行起点 */
    float *Cl2 = C;            /* 写回: 低 16 行起点 */
    float *Ch2 = C + 16 * 32;  /* 写回: 高 16 行起点 */
    const float *A1 = A + 16;  /* A 列后 16 行 */
    const float *B1 = B + 16;  /* B 行后 16 列 */
    __asm__ volatile(
        "smstart sm\n\t"
        "smstart za\n\t"
        "ptrue p0.s, all\n\t"
        /* 加载 C 初值进 4 tile (完整 C += A*B 语义) */
        "mov w12, #0\n\t"
        "3:\n\t"
        "ld1w {za0h.s[w12, 0]}, p0/z, [%[Cl]]\n\t"
        "add %[Cl], %[Cl], #64\n\t"
        "ld1w {za1h.s[w12, 0]}, p0/z, [%[Cl]]\n\t"
        "add %[Cl], %[Cl], #64\n\t"
        "ld1w {za2h.s[w12, 0]}, p0/z, [%[Ch]]\n\t"
        "add %[Ch], %[Ch], #64\n\t"
        "ld1w {za3h.s[w12, 0]}, p0/z, [%[Ch]]\n\t"
        "add %[Ch], %[Ch], #64\n\t"
        "add w12, w12, #1\n\t"
        "cmp w12, #16\n\t"
        "b.ne 3b\n\t"
        /* K 循环 (论文 Fig.7): 每步 2 A 向量 + 2 B 向量, 4 fmopa */
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
        /* 写回: za0-za3 的 16 行各存到 C (行距 128 字节) */
        "mov w12, #0\n\t"
        "2:\n\t"
        "st1w {za0h.s[w12, 0]}, p0, [%[Cl2]]\n\t"
        "add %[Cl2], %[Cl2], #64\n\t"
        "st1w {za1h.s[w12, 0]}, p0, [%[Cl2]]\n\t"
        "add %[Cl2], %[Cl2], #64\n\t"
        "st1w {za2h.s[w12, 0]}, p0, [%[Ch2]]\n\t"
        "add %[Ch2], %[Ch2], #64\n\t"
        "st1w {za3h.s[w12, 0]}, p0, [%[Ch2]]\n\t"
        "add %[Ch2], %[Ch2], #64\n\t"
        "add w12, w12, #1\n\t"
        "cmp w12, #16\n\t"
        "b.ne 2b\n\t"
        "smstop za\n\t"
        "smstop sm\n\t"
        : [A] "+r" (A), [A1] "+r" (A1), [B] "+r" (B), [B1] "+r" (B1),
          [Cl] "+r" (Cl), [Ch] "+r" (Ch), [Cl2] "+r" (Cl2), [Ch2] "+r" (Ch2),
          [K] "+r" (K)
        :
        : "memory", "w12"
    );
}

/* KirbyMM GEMM 入口 (论文 §IV-A + §V MacroKernel)
 * 当前: 32×32 块遍历 + pack A/B/C (简单实现, 后续按 §V-A/§V-B 优化)
 * M=N=16: 走 16×16 单瓦片 SME (验证用)
 * M,N 都是 32 倍数: 32×32 块遍历, 每块走 SME
 * 其他 (含边界): fallback naive
 * 完整 C += A*B 语义
 */
void kirby_sgemm_fp32(int M, int N, int K,
                      const float *A, const float *B, float *C) {
    /* M=N=16 单块 SME 路径 (小规模验证) */
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
     * 每块 pack A (转置 col-major) + pack B (跨 stride row-major) + pack/unpack C
     * (后续 §V-A 求解 blkm/blkn/blkk, §V-B ZA-tile 隐式转置优化 packing) */
    float *A_blk = (float *)malloc(sizeof(float) * 32 * (size_t)K);
    float *B_blk = (float *)malloc(sizeof(float) * (size_t)K * 32);
    float *C_blk = (float *)malloc(sizeof(float) * 32 * 32);

    for (int ii = 0; ii < M; ii += 32) {
        for (int jj = 0; jj < N; jj += 32) {
            /* pack A[ii:ii+32, :] → col-major 32×K (转置) */
            for (int k = 0; k < K; k++)
                for (int i = 0; i < 32; i++)
                    A_blk[k * 32 + i] = A[(ii + i) * K + k];
            /* pack B[:, jj:jj+32] → row-major K×32 (跨 stride) */
            for (int k = 0; k < K; k++)
                for (int j = 0; j < 32; j++)
                    B_blk[k * 32 + j] = B[k * N + (jj + j)];
            /* pack C[ii:ii+32, jj:jj+32] → 连续 32×32 (加载初值) */
            for (int i = 0; i < 32; i++)
                memcpy(C_blk + i * 32, C + (ii + i) * N + jj, 32 * sizeof(float));
            /* microkernel: C_blk += A_blk * B_blk */
            micro_32x32_kupdate(C_blk, A_blk, B_blk, K);
            /* unpack C_blk → 原 C */
            for (int i = 0; i < 32; i++)
                memcpy(C + (ii + i) * N + jj, C_blk + i * 32, 32 * sizeof(float));
        }
    }

    free(A_blk); free(B_blk); free(C_blk);
}
