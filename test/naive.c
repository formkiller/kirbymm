/*
 * naive.c — naive FP32 GEMM 实现 (正确性基准)
 *
 * 语义: C += A * B  (BLAS 风格累加)
 * 存储: row-major
 *       A: M x K, B: K x N, C: M x N
 *
 * 用 restrict 提示编译器 A/B/C 不别名, -O3 下可自动向量化,
 * 但本质仍是 ijk 三重循环, 作为后续优化的正确性与性能下限基准.
 */
#include "kirby.h"

void naive_sgemm_fp32(int M, int N, int K,
                      const float * restrict A,
                      const float * restrict B,
                      float * restrict C) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] += acc;
        }
    }
}
