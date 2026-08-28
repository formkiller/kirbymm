/*
 * kirby.h — KirbyMM 公共接口
 *
 * 约定: 所有 GEMM 函数采用 BLAS 风格的累加语义 C += A * B
 *       矩阵存储为 row-major, FP32
 *       A: M x K, B: K x N, C: M x N
 */
#ifndef KIRBY_H
#define KIRBY_H

#ifdef __cplusplus
extern "C" {
#endif

/* naive GEMM (纯 C 三重循环, 正确性基准, 无优化) */
void naive_sgemm_fp32(int M, int N, int K,
                      const float *A, const float *B, float *C);

/* KirbyMM GEMM (待实现, 阶段一主例程) */
void kirby_sgemm_fp32(int M, int N, int K,
                      const float *A, const float *B, float *C);

/* [仅测试用] microkernel 裸性能探针: 直通 micro_32x32_kupdate
 * 不属于论文复现范围, 供 tools/kernel_bench.c 定标 kernel 峰值 */
void kirby_kernel_probe(float *C, int ldc, const float *A, const float *B, int K);

#ifdef __cplusplus
}
#endif

#endif /* KIRBY_H */
