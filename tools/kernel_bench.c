/*
 * kernel_bench.c — microkernel 裸性能定标 (不含任何调度层开销)
 *
 * 目的:
 *   1. 测出 M4 上 micro_32x32_kupdate 的纯 kernel GFLOPS
 *   2. 对照 ldc=32 (连续块, 等价打包版布局) vs ldc=1024 (直连大矩阵行距)
 *      —— 归因直连 C 的收益来源: 是省了 pack, 还是 ldc 路径本身更快?
 *   3. 定标 M4 真实 SME 峰值, 修正 bench.c 的 "512 GFLOPS" 口径
 *
 * 方法: 预分配 A/B/C 缓冲, kernel 重复调用 K 次 (C 累加语义天然支持重复调用,
 *       每次 C += A*B, 无需重置 C), 计时 / FLOPs
 * 注意: 重复调用时 C 值持续增长, FP32 溢出会影响 fmla? fmopa 累加也可能
 *       因数值变大而变慢 (denormal/inf) —— 用小值 A/B 控制增长幅度
 */
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

/* 与 primary.c 内部一致的 kernel 声明 (通过 header 暴露) */
#include "kirby.h"

/* kernel_bench 用: 直接压 kirby_sgemm_fp32 的 32x32 单块路径?
 * 不行 — kirby 入口对 32x32xK 走五层循环. 方案: 在 kirby.h 增加
 * kernel 级测试入口, 由 primary.c 导出一个薄包装 kirby_kernel_probe
 * (在 primary.c 末尾添加, 仅测试用, 不属于论文复现范围) */
void kirby_kernel_probe(float *C, int ldc, const float *A, const float *B, int K);

static float *A32, *B32, *C32;

static void bench_case(const char *name, int ldc, int K, int reps) {
    /* 预热 */
    kirby_kernel_probe(C32, ldc, A32, B32, K);
    double t0 = now_sec();
    for (int r = 0; r < reps; r++)
        kirby_kernel_probe(C32, ldc, A32, B32, K);
    double dt = now_sec() - t0;

    double flops = 2.0 * 32.0 * 32.0 * K * reps;
    printf("  [%-28s] ldc=%4d K=%4d reps=%5d  %8.2f GFLOPS\n",
           name, ldc, K, reps, flops / dt / 1e9);
}

int main(void) {
    printf("=== microkernel-only benchmark (scheduling excluded) ===\n");
    /* A/B 用小值抑制 C 溢出: K*reps*val^2 ~ 1e4 量级 */
    A32 = malloc(sizeof(float) * 32 * 256);
    B32 = malloc(sizeof(float) * 256 * 32);
    C32 = malloc(sizeof(float) * 1024 * 1024);
    for (int i = 0; i < 32 * 256; i++)  A32[i] = 0.01f * ((i % 7) - 3);
    for (int i = 0; i < 256 * 32; i++)  B32[i] = 0.01f * ((i % 5) - 2);
    memset(C32, 0, sizeof(float) * 1024 * 1024);

    /* (a) ldc=32: 连续 32x32 块, 等价打包版布局 */
    bench_case("packed-layout (ldc=32)", 32, 128, 2000);
    /* (b) ldc=1024: 直连 1024 大矩阵的行距 (离散写回) */
    bench_case("direct-C (ldc=1024)", 1024, 128, 2000);
    /* (c) ldc=512: 中等行距 */
    bench_case("direct-C (ldc=512)", 512, 128, 2000);
    /* (d) ldc=32 K=512: 看 K 变长的吞吐 (更长 K 循环摊薄载入写回) */
    bench_case("packed-layout (ldc=32) K=512", 32, 512, 500);

    printf("\n(对比 bench.c 整体数字: 整体 GFLOPS / kernel GFLOPS = 调度效率)\n");
    printf("(M4 真实 SME 峰值 ~ kernel GFLOPS 上限; 以此修正 bench.c 口径)\n");
    free(A32); free(B32); free(C32);
    return 0;
}
