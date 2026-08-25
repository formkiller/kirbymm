/*
 * verify.c — KirbyMM 正确性验证框架
 *
 * 流程:
 *   1. 生成随机矩阵 A, B (值域 [-1, 1])
 *   2. C 从零开始, 调用 ref 得 C_ref, 调用 test 得 C_test
 *   3. 逐元素比对 max abs error, 阈值 1e-3 (FP32 GEMM 累积误差容差)
 *
 * 当前: ref = test = naive_sgemm (自对照, 验证框架本身)
 * 后续: test 换成 kirby_sgemm_fp32 即可对照 KirbyMM 实现
 *
 * 测试用例覆盖:
 *   - 对齐规模 (64/128/256): 主例程目标场景
 *   - 边界规模 (35x32x32): Fig.8 专用例程场景
 *   - 小边界 (17x16x16): 专用例程边界
 *   - 非对齐 (48x48x48): 通用边界
 */
#include "kirby.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* 函数指针类型: GEMM */
typedef void (*gemm_fn_t)(int, int, int, const float *, const float *, float *);

static float *alloc_mat(int rows, int cols) {
    float *p = malloc(sizeof(float) * (size_t)rows * (size_t)cols);
    if (!p) { fprintf(stderr, "alloc failed %dx%d\n", rows, cols); exit(1); }
    return p;
}

static void fill_random(float *m, int rows, int cols) {
    /* 确定性种子, 保证可复现 */
    static unsigned int seed = 12345u;
    for (int i = 0; i < rows * cols; i++) {
        seed = seed * 1103515245u + 12345u;
        int r = (int)((seed >> 16) & 0x7fff) % 2001 - 1000;  /* [-1000, 1000] */
        m[i] = (float)r / 1000.0f;  /* [-1, 1] */
    }
}

static void zero_mat(float *m, int rows, int cols) {
    memset(m, 0, sizeof(float) * (size_t)rows * (size_t)cols);
}

/* 比对两份 C, 返回 max abs error */
static float compare_max_err(const float *a, const float *b, int M, int N) {
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > max_err) max_err = e;
    }
    return max_err;
}

/* 单组测试用例: 返回 1=PASS, 0=FAIL */
static int run_case(int M, int N, int K,
                    gemm_fn_t ref, gemm_fn_t test) {
    float *A      = alloc_mat(M, K);
    float *B      = alloc_mat(K, N);
    float *C_ref  = alloc_mat(M, N);
    float *C_test = alloc_mat(M, N);

    fill_random(A, M, K);
    fill_random(B, K, N);
    zero_mat(C_ref, M, N);
    zero_mat(C_test, M, N);

    ref(M, N, K, A, B, C_ref);
    test(M, N, K, A, B, C_test);

    float err = compare_max_err(C_ref, C_test, M, N);
    /* 阈值 1e-3: FP32 GEMM 累积误差容差, 对 K<=256 足够保守 */
    const float eps = 1e-3f;
    int pass = (err < eps);

    printf("  [%4dx%4dx%4d]  max_err=%8.2e  eps=%8.2e  %s\n",
           M, N, K, err, eps, pass ? "PASS" : "FAIL");

    free(A); free(B); free(C_ref); free(C_test);
    return pass;
}

int main(void) {
    printf("=== KirbyMM correctness verification ===\n\n");

    /* ref = naive, test = naive (自对照, 验证框架) */
    gemm_fn_t ref  = naive_sgemm_fp32;
    gemm_fn_t test = naive_sgemm_fp32;

    /* 测试用例集 */
    struct { int M, N, K; const char *note; } cases[] = {
        { 64,  64,  64, "aligned"      },
        {128, 128, 128, "aligned"      },
        {256, 256, 256, "aligned"      },
        { 35,  32,  32, "edge (Fig.8)" },
        { 17,  16,  16, "small edge"   },
        { 48,  48,  48, "unaligned"    },
        {1024,1024,1024, "large"       },
    };
    int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    printf("[self-check] naive vs naive (verify framework):\n");
    int all_pass = 1;
    for (int i = 0; i < n_cases; i++) {
        int ok = run_case(cases[i].M, cases[i].N, cases[i].K, ref, test);
        if (!ok) all_pass = 0;
    }

    printf("\n=== result: %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");

    /* 后续接入 KirbyMM 时:
     *   gemm_fn_t test = kirby_sgemm_fp32;
     *   重新编译链接, 即可对照 KirbyMM vs naive
     */
    printf("\n(note: currently naive vs naive self-check.\n"
           " to verify KirbyMM: change test fn to kirby_sgemm_fp32)\n");

    return all_pass ? 0 : 1;
}
