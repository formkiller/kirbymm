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

/* ZA-tile 隐式转置打包 (论文 §V-B, Fig.10)
 * 把 row-major 16×16 子块转置成 col-major 16×16
 * 流程:
 *   1. 16 行 ld1w 加载进 za0 (水平 slice = 行, 连续读 src 行)
 *   2. 16 列 st1w 从 za0 读出 (垂直 slice = 列, 连续写 dst 列)
 * src: row-major 16×16, 行距 src_stride (元素)
 * dst: col-major 16×16, 列距 16 (元素)
 */
__attribute__((target("sve,sme")))
static void za_transpose_16x16(float *dst, int dst_stride,
                                const float *src, int src_stride) {
    float *s = (float *)src;
    float *d = dst;
    long src_skip = (long)(src_stride - 16) * 4;  /* 行间跳距: src 下一行 */
    long dst_skip = (long)(dst_stride - 16) * 4;  /* 列间跳距: dst 下一列 */
    __asm__ volatile(
        "smstart sm\n\t"
        "smstart za\n\t"
        "ptrue p0.s, all\n\t"
        /* 16 行加载进 za0 (水平 slice, 连续读 src 行) */
        "mov w12, #0\n\t"
        "1:\n\t"
        "ld1w {za0h.s[w12, 0]}, p0/z, [%[s]]\n\t"
        "add %[s], %[s], %[ss]\n\t"
        "add w12, w12, #1\n\t"
        "cmp w12, #16\n\t"
        "b.ne 1b\n\t"
        /* 16 列从 za0 读出到 dst (垂直 slice, 连续写 dst 列) */
        "mov w12, #0\n\t"
        "2:\n\t"
        "st1w {za0v.s[w12, 0]}, p0, [%[d]]\n\t"
        "add %[d], %[d], %[ds]\n\t"
        "add w12, w12, #1\n\t"
        "cmp w12, #16\n\t"
        "b.ne 2b\n\t"
        "smstop za\n\t"
        "smstop sm\n\t"
        : [s] "+r" (s), [d] "+r" (d)
        : [ss] "r" (src_skip), [ds] "r" (dst_skip)
        : "memory", "w12"
    );
}

/* 专用例程 microkernel (论文 §IV-B, Fig.8: M=35 直接复现)
 * M=35: 主体 32×32 走 4 fmopa (za0-za3), 边缘 3 行走 6 fmla (z10-z15)
 * C: 35×32 row-major (行距 32 = 128 字节)
 * A: col-major 35×K (每列 35 FP32 连续, 列距 35)
 * B: row-major K×32 (每行 32 FP32 连续)
 *
 * K 循环每步 (论文 Fig.8(b)):
 *   4 fmopa (主体 32×32) + 3×2 fmla (边缘 3×32, A 标量广播 × B 向量)
 *   M4 上 fmopa/fmla 共享 SME 单元不 overlap, 但代码结构与论文一致
 *   LX2 上 overlap 使 T_{4mopa+6mla} = T_{4mopa} (Eq.4)
 */
__attribute__((target("sve,sme")))
static void micro_dedicated_35x32(float *C, const float *A, const float *B, int K) {
    float *Cl = C;               /* C 主体低 16 行 */
    float *Ch = C + 16 * 32;     /* C 主体高 16 行 */
    float *Cl2 = C;              /* 写回低 16 行 */
    float *Ch2 = C + 16 * 32;    /* 写回高 16 行 */
    float *Ce = C + 32 * 32;     /* C 边缘 3 行起点 */
    float *Ce2 = Ce;             /* 写回边缘 */
    const float *A1 = A + 16;     /* A 列行 16-31 */
    const float *A_rem = A + 32; /* A 列行 32-34 (边缘 3 标量) */
    const float *B1 = B + 16;    /* B 行后 16 列 */
    __asm__ volatile(
        "smstart sm\n\t"
        "smstart za\n\t"
        "ptrue p0.s, all\n\t"
        /* 加载 C 初值: za0-za3 (主体 32×32) + z10-z15 (边缘 3×32) */
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
        /* 加载边缘 C 初值 (3 行 × 2 向量 = 6 次 ld1w) */
        "ld1w z10.s, p0/z, [%[Ce]]\n\t"
        "add %[Ce], %[Ce], #64\n\t"
        "ld1w z11.s, p0/z, [%[Ce]]\n\t"
        "add %[Ce], %[Ce], #64\n\t"     /* Ce 跳到行 33 */
        "ld1w z12.s, p0/z, [%[Ce]]\n\t"
        "add %[Ce], %[Ce], #64\n\t"
        "ld1w z13.s, p0/z, [%[Ce]]\n\t"
        "add %[Ce], %[Ce], #64\n\t"     /* Ce 跳到行 34 */
        "ld1w z14.s, p0/z, [%[Ce]]\n\t"
        "add %[Ce], %[Ce], #64\n\t"
        "ld1w z15.s, p0/z, [%[Ce]]\n\t"
        /* K 循环: 4 fmopa (主体) + 6 fmla (边缘, 3 标量 × 2 B 向量) */
        "1:\n\t"
        "ld1w z0.s, p0/z, [%[A]]\n\t"          /* A 列行 0-15 */
        "ld1w z1.s, p0/z, [%[A1]]\n\t"         /* A 列行 16-31 */
        "ld1w z2.s, p0/z, [%[B]]\n\t"          /* B 行前 16 列 */
        "ld1w z3.s, p0/z, [%[B1]]\n\t"         /* B 行后 16 列 */
        "fmopa za0.s, p0/m, p0/m, z0.s, z2.s\n\t"
        "fmopa za1.s, p0/m, p0/m, z0.s, z3.s\n\t"
        "fmopa za2.s, p0/m, p0/m, z1.s, z2.s\n\t"
        "fmopa za3.s, p0/m, p0/m, z1.s, z3.s\n\t"
        /* 边缘: A 标量 (行 32/33/34) 广播 × B 向量, fmla 累加进 z10-z15 */
        "ldr w0, [%[Ar]]\n\t"                  /* A[32, k] */
        "dup z4.s, w0\n\t"
        "fmla z10.s, p0/m, z4.s, z2.s\n\t"    /* C[32, 0:16] += A[32] ⊗ B[0:16] */
        "fmla z11.s, p0/m, z4.s, z3.s\n\t"    /* C[32,16:32] */
        "ldr w0, [%[Ar], #4]\n\t"              /* A[33, k] */
        "dup z5.s, w0\n\t"
        "fmla z12.s, p0/m, z5.s, z2.s\n\t"     /* C[33, 0:16] */
        "fmla z13.s, p0/m, z5.s, z3.s\n\t"     /* C[33,16:32] */
        "ldr w0, [%[Ar], #8]\n\t"              /* A[34, k] */
        "dup z6.s, w0\n\t"
        "fmla z14.s, p0/m, z6.s, z2.s\n\t"     /* C[34, 0:16] */
        "fmla z15.s, p0/m, z6.s, z3.s\n\t"     /* C[34,16:32] */
        /* 指针推进: A 列距=35 (35*4=140), B 行距=32 (128) */
        "add %[A],  %[A],  #140\n\t"
        "add %[A1], %[A1], #140\n\t"
        "add %[Ar], %[Ar], #140\n\t"
        "add %[B],  %[B],  #128\n\t"
        "add %[B1], %[B1], #128\n\t"
        "subs %w[K], %w[K], #1\n\t"
        "b.ne 1b\n\t"
        /* 写回主体 za0-za3 (32×32, 与 micro_32x32 相同) */
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
        /* 写回边缘 z10-z15 (3×32, 6 次 st1w) */
        "st1w z10.s, p0, [%[Ce2]]\n\t"
        "add %[Ce2], %[Ce2], #64\n\t"
        "st1w z11.s, p0, [%[Ce2]]\n\t"
        "add %[Ce2], %[Ce2], #64\n\t"
        "st1w z12.s, p0, [%[Ce2]]\n\t"
        "add %[Ce2], %[Ce2], #64\n\t"
        "st1w z13.s, p0, [%[Ce2]]\n\t"
        "add %[Ce2], %[Ce2], #64\n\t"
        "st1w z14.s, p0, [%[Ce2]]\n\t"
        "add %[Ce2], %[Ce2], #64\n\t"
        "st1w z15.s, p0, [%[Ce2]]\n\t"
        "smstop za\n\t"
        "smstop sm\n\t"
        : [A] "+r" (A), [A1] "+r" (A1), [Ar] "+r" (A_rem),
          [B] "+r" (B), [B1] "+r" (B1),
          [Cl] "+r" (Cl), [Ch] "+r" (Ch), [Cl2] "+r" (Cl2), [Ch2] "+r" (Ch2),
          [Ce] "+r" (Ce), [Ce2] "+r" (Ce2), [K] "+r" (K)
        :
        : "memory", "w12", "w0", "z0", "z1", "z2", "z3",
          "z4", "z5", "z6", "z10", "z11", "z12", "z13", "z14", "z15"
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

/* KirbyMM GEMM 入口 (论文 §IV-A + §V-A 严格复现)
 * Goto 5 层循环 (算法 1), 参数由 Eq.5/6/7 联立求解:
 *   blkm=blkn=128, blkk=128, si=sj=32 (microkernel 块)
 *
 * 循环结构:
 *   L1 (j,  blkn=128): C 沿 N 顶层分块
 *   L2 (k,  blkk=128): K 分块, B 面板 pack 驻留 L2
 *   L3 (i,  blkm=128): M 分块, A 块 pack 驻留 L2
 *   L4 (jj, sj=32):   blkn 内 N 再分块 (L2 驻留)
 *   L5 (ii, si=32):   blkm 内 M 再分块 = microkernel (L1 驻留)
 *
 * Packing (论文 §V, 当前简单实现, §V-B 将用 ZA-tile 优化):
 *   A_block: blkm×blkk, 按 32 行子块组织, 每子块 col-major 32×blkk (列距 32)
 *   B_panel: blkk×blkn, 按 32 列子块组织, 每子块 row-major blkk×32 (行距 32)
 *   C_blk:   32×32 连续, pack/unpack microkernel 用
 *
 * M=N=16: 走 16×16 单瓦片 SME (验证用)
 * M,N 是 32 倍数: 走 5 层循环 (blkm/blkn 自适应 ≤128)
 * 其他: fallback naive
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

    /* M=35, N=32: 走专用例程 (§IV-B, Fig.8 的 fmopa+fmla 混合) */
    if (M == 35 && N == 32) {
        float *A_col = (float *)malloc(sizeof(float) * 35 * (size_t)K);
        for (int k = 0; k < K; k++)
            for (int i = 0; i < 35; i++)
                A_col[k * 35 + i] = A[i * K + k];
        micro_dedicated_35x32(C, A_col, B, K);
        free(A_col);
        return;
    }

    /* 非 32 倍数 → naive 兜底 */
    if (M % 32 != 0 || N % 32 != 0) {
        naive_sgemm_fp32(M, N, K, A, B, C);
        return;
    }

    const int BLKM = 128, BLKN = 128, BLKK = 128, SI = 32, SJ = 32;

    /* 预分配 packing 缓冲 (一次, 全程复用) */
    float *A_block = (float *)malloc(sizeof(float) * BLKM * BLKK);  /* 128×128 */
    float *B_panel = (float *)malloc(sizeof(float) * BLKK * BLKN);  /* 128×128 */
    float *C_blk   = (float *)malloc(sizeof(float) * SI * SJ);      /* 32×32 */

    /* L1: j 循环 (blkn) */
    for (int j = 0; j < N; j += BLKN) {
        int blkn = (j + BLKN <= N) ? BLKN : (N - j);

        /* L2: k 循环 (blkk) */
        for (int k = 0; k < K; k += BLKK) {
            int blkk = (k + BLKK <= K) ? BLKK : (K - k);

            /* pack B[k:k+blkk, j:j+blkn] → B_panel
             * 按 32 列子块组织, 每子块 row-major blkk×32 (行距 32) */
            int n_blocks = (blkn + SJ - 1) / SJ;
            for (int jb = 0; jb < n_blocks; jb++) {
                int sj_cur = (jb * SJ + SJ <= blkn) ? SJ : (blkn - jb * SJ);
                for (int kk = 0; kk < blkk; kk++)
                    for (int jj = 0; jj < sj_cur; jj++)
                        B_panel[(size_t)jb * BLKK * SJ + (size_t)kk * SJ + jj] =
                            B[(size_t)(k + kk) * N + (j + jb * SJ + jj)];
            }

            /* L3: i 循环 (blkm) */
            for (int i = 0; i < M; i += BLKM) {
                int blkm = (i + BLKM <= M) ? BLKM : (M - i);

                /* pack A[i:i+blkm, k:k+blkk] → A_block (§V-B ZA-tile 隐式转置)
                 * 32×blkk 分成 2×(blkk/16) 个 16×16 子块, 每个 ZA tile 转置
                 * src: A row-major 16×16, 行距 K (连续读行)
                 * dst: A_block col-major 16×16, 列距 32 (连续写列) */
                int m_blocks = (blkm + SI - 1) / SI;
                for (int ib = 0; ib < m_blocks; ib++) {
                    int si_cur = (ib * SI + SI <= blkm) ? SI : (blkm - ib * SI);
                    for (int bi = 0; bi < si_cur / 16; bi++) {
                        for (int bk = 0; bk + 16 <= blkk; bk += 16) {
                            za_transpose_16x16(
                                A_block + (size_t)ib * BLKK * SI + (size_t)bk * SI + bi * 16,
                                SI,   /* dst_stride = 32 (A_block col-major 列距) */
                                A + (size_t)(i + ib * SI + bi * 16) * K + (k + bk),
                                K     /* src_stride = K (A row-major 行距) */
                            );
                        }
                        /* 余数列 (blkk 非 16 倍数), 简单循环兜底 */
                        int rem = blkk % 16;
                        if (rem > 0) {
                            int bk = blkk - rem;
                            for (int kk = 0; kk < rem; kk++)
                                for (int ii = 0; ii < 16; ii++)
                                    A_block[(size_t)ib * BLKK * SI + (size_t)(bk + kk) * SI + bi * 16 + ii] =
                                        A[(size_t)(i + ib * SI + bi * 16 + ii) * K + (k + bk + kk)];
                        }
                    }
                }

                /* L4: jj 循环 (sj=32) */
                for (int jj = 0; jj < blkn; jj += SJ) {
                    int sj_cur = (jj + SJ <= blkn) ? SJ : (blkn - jj);
                    int jb = jj / SJ;

                    /* L5: ii 循环 (si=32) = microkernel */
                    for (int ii = 0; ii < blkm; ii += SI) {
                        int si_cur = (ii + SI <= blkm) ? SI : (blkm - ii);
                        int ib = ii / SI;

                        /* pack C[i+ii:i+ii+si, j+jj:j+jj+sj] → C_blk (32×32 连续) */
                        for (int r = 0; r < si_cur; r++)
                            memcpy(C_blk + r * 32,
                                   C + (size_t)(i + ii + r) * N + (j + jj),
                                   sj_cur * sizeof(float));

                        /* microkernel: C_blk += A_sub * B_sub (完整 C += A*B) */
                        micro_32x32_kupdate(C_blk,
                            A_block + (size_t)ib * BLKK * SI,
                            B_panel + (size_t)jb * BLKK * SJ,
                            blkk);

                        /* unpack C_blk → 原 C */
                        for (int r = 0; r < si_cur; r++)
                            memcpy(C + (size_t)(i + ii + r) * N + (j + jj),
                                   C_blk + r * 32,
                                   sj_cur * sizeof(float));
                    }
                }
            }
        }
    }

    free(A_block); free(B_panel); free(C_blk);
}
