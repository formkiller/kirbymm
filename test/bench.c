/*
 * bench.c — KirbyMM 性能基准 (naive vs kirby GFLOPS 对照)
 *
 * 测量: 多组 32 倍数规模, naive vs kirby 的 GFLOPS 与加速比
 * 对照: M4 SME 峰值 ~512 GFLOPS (0.512 TFLOPS, FP32 单核)
 *
 * 注意: 当前 MacroKernel 每块有 pack/unpack malloc+memcpy 开销,
 *       大概率 kirby 比 naive 慢 — bench 会揭示瓶颈, 指导后续优化
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
    printf("M4 SME peak: ~512 GFLOPS (FP32, 0.512 TFLOPS)\n\n");
    printf("%-10s %10s %10s %10s %10s\n", "size", "naive", "kirby", "speedup", "kirby/peak");
    printf("------------------------------------------------------------\n");

    int sizes[] = {32, 64, 128, 256, 512, 1024};
    int n = (int)(sizeof(sizes) / sizeof(sizes[0]));
    for (int i = 0; i < n; i++) {
        int s = sizes[i];
        double gn = bench_gflops(naive_sgemm_fp32, s, s, s);
        double gk = bench_gflops(kirby_sgemm_fp32, s, s, s);
        printf("%-10d %10.2f %10.2f %9.2fx %9.1f%%\n",
               s, gn, gk, gk / gn, gk / 512.0 * 100);
    }

    printf("\n(note: current MacroKernel has pack/unpack overhead per block,\n");
    printf(" kirby may be slower than naive — this is expected pre-optimization.\n");
    printf(" next step: pre-allocated packing buffers + ZA-tile packing.)\n");
    return 0;
}
