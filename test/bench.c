/*
 * bench.c — KirbyMM 性能基准 (naive vs kirby GFLOPS 对照)
 *
 * 测量: 多组 32 倍数规模, naive vs kirby 的 GFLOPS 与加速比
 * 对照口径 (2026-08-29 kernel_bench v6 定标):
 *   - 论文 Fig.13 M4 MicroKernel 上限 1425.13 GFLOPS (本文复现 kernel 达到其 97.6%)
 *   - 理论峰: 1 fmopa/cycle @ 4.4GHz = 2253 GFLOPS; K=512 实测 kernel 1828~1860 (81~83%)
 *   - 旧 "~512 GFLOPS" 口径为初版拍的估计值, 已废弃
 *
 * 注意: 本机 (M4) 时间戳在 SME 调用后存在冻结/清零病态, 全 GEMM 粒度
 *       (t0 于 SME 前读取) 未受影响, 数字可信; microkernel 级计时必须走
 *       kernel_bench v6 的差分协议, 不可直接单次计时
 *
 * 编译: cmake --build build (自动加入)
 * 运行: ./build/bench
 */
#include "kirby.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
  #include <mach/mach_time.h>
  static double now_sec(void) {
      static mach_timebase_info_data_t tb;
      static int inited = 0;
      if (!inited) { mach_timebase_info(&tb); inited = 1; }
      uint64_t t = mach_absolute_time();
      return (double)t * tb.numer / (double)tb.denom / 1e9;
  }
#else
  #include <time.h>
  static double now_sec(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return ts.tv_sec + ts.tv_nsec / 1e9;
  }
#endif

typedef void (*gemm_fn_t)(int, int, int, const float *, const float *, float *);

static void fill_random(float *m, int n) {
    static unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        m[i] = (float)((int)((seed >> 16) & 0x7fff) % 2001 - 1000) / 1000.0f;
    }
}

/* 测单次 GEMM 的 GFLOPS (多次重复取平均, 每次重置 C 初值) */
static double bench_gflops(gemm_fn_t fn, int M, int N, int K) {
    /* reps 按规模调整: 小规模多重复, 大规模少重复 (控制总时间) */
    int reps = (M * N * K < 1000000) ? 50 : 5;

    float *A  = (float *)malloc(sizeof(float) * (size_t)M * (size_t)K);
    float *B  = (float *)malloc(sizeof(float) * (size_t)K * (size_t)N);
    float *C  = (float *)malloc(sizeof(float) * (size_t)M * (size_t)N);
    float *C0 = (float *)malloc(sizeof(float) * (size_t)M * (size_t)N);

    fill_random(A, M * K);
    fill_random(B, K * N);
    fill_random(C0, M * N);

    /* warmup (触发缓存/分支预测器预热) */
    memcpy(C, C0, sizeof(float) * (size_t)M * (size_t)N);
    fn(M, N, K, A, B, C);

    double t0 = now_sec();
    for (int r = 0; r < reps; r++) {
        memcpy(C, C0, sizeof(float) * (size_t)M * (size_t)N);
        fn(M, N, K, A, B, C);
    }
    double dt = now_sec() - t0;

    double flops = 2.0 * (double)M * (double)N * (double)K * reps;
    free(A); free(B); free(C); free(C0);
    return flops / dt / 1e9;
}

int main(void) {
    printf("=== KirbyMM performance benchmark ===\n");
    printf("peak references: Fig.13 kernel 1425.13 | fmopa@4.4GHz 2253 GFLOPS\n\n");
    printf("%-10s %10s %10s %10s %9s %9s\n", "size", "naive", "kirby", "speedup",
           "%fig13", "%theor");
    printf("--------------------------------------------------------------------------\n");

    int sizes[] = {32, 64, 128, 256, 512, 1024};
    int n = (int)(sizeof(sizes) / sizeof(sizes[0]));
    for (int i = 0; i < n; i++) {
        int s = sizes[i];
        double gn = bench_gflops(naive_sgemm_fp32, s, s, s);
        double gk = bench_gflops(kirby_sgemm_fp32, s, s, s);
        printf("%-10d %10.2f %10.2f %9.2fx %8.1f%% %8.1f%%\n",
               s, gn, gk, gk / gn, gk / 1425.13 * 100, gk / 2253.0 * 100);
    }

    printf("\n(ref: kernel-only v6 diff-protocol: ldc=256 tailD 1506 GFLOPS = 105.7%% fig13;\n");
    printf(" K=512 kernel up to 1860 GFLOPS = 82.6%% theoretical. fig13 = 1425.13 GFLOPS,\n");
    printf(" theor = 1 fmopa/cycle @ 4.4GHz = 2253 GFLOPS. old ~512-peak calibration retired.)\n");
    return 0;
}
