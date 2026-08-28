/*
 * kernel_bench.c v3 — microkernel 裸性能定标 (自诊断版)
 *
 * 背景: v2 (85ffebb) 在 M4 上 7 用例全部打印 0.00, 且运行在数分钟内完成.
 * 时间账反证: "0.00" 要求 best < 0.005 GFLOPS, 即每一轮 dt > 105 s,
 *   343 轮合计 > 85 min — 与实际运行时长矛盾. 唯一自洽解释:
 *   (a) 每轮 g 均为 NaN (NaN 不满足 "> 0", best 恒为初值 0.0) — 计时函数异常
 *   (b) M4 侧编译源与本仓库 v2 不一致 (手动同步分歧)
 * v3 对策:
 *   1. 版本横幅 "v3": 输出无 "v3" 字样 → 坐实 (b), 先修同步
 *   2. 计时统一切换 clock_gettime(CLOCK_MONOTONIC) (macOS 10.12+/Linux 同路径),
 *      彻底移除 mach_timebase 变量 — 若系 (a) 的 timebase 异常, 此改动直接修复
 *   3. 每用例打印 round0 的 dt; best 无效 (NaN/inf/<=0) 时打印 t0/t1/dt 原始值
 *   4. main 前置探针接线自检 (probe vs naive, ldc=32/K=64), 防止"测的是空转"
 *
 * 方法: 预分配 A/B/C, kernel 重复调用 (C += A*B 累加语义), best-of-7 取最快轮
 * 数值安全: K=512, reps=500 时 C 元素 ~ N(0, 20^2), 远离溢出/denormal
 * v2 修复保留: A32/B32 按 K 上限 1024 分配 (v1 在 K=512 用例越界读堆)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* v3: 单一计时路径, 无平台分叉 (M4 的 macOS 15 对 CLOCK_MONOTONIC 支持完备) */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* 与 primary.c 内部一致的 kernel 声明 (通过 header 暴露) */
#include "kirby.h"

/* [仅测试用] 探针: primary.c 导出, 直通 micro_32x32_kupdate */
void kirby_kernel_probe(float *C, int ldc, const float *A, const float *B, int K);

static float *A32, *B32, *C32;

/* 单轮计时: kernel 重复调用 reps 次; 回传 t0/t1/dt 供诊断 */
static double bench_once(int ldc, int K, int reps,
                         double *dt_out, double *t0_out, double *t1_out) {
    kirby_kernel_probe(C32, ldc, A32, B32, K);   /* 预热 */
    double t0 = now_sec();
    for (int r = 0; r < reps; r++)
        kirby_kernel_probe(C32, ldc, A32, B32, K);
    double t1 = now_sec();
    double dt = t1 - t0;
    if (dt_out) *dt_out = dt;
    if (t0_out) *t0_out = t0;
    if (t1_out) *t1_out = t1;
    double flops = 2.0 * 32.0 * 32.0 * K * reps;
    return flops / dt / 1e9;
}

static int any_invalid = 0;

/* best-of-7: 峰值定标取最快轮 (OS 抖动/降频轮直接淘汰) */
static void bench_case(const char *name, int ldc, int K, int reps) {
    double best = 0.0, dt0 = -1.0, ta0 = 0.0, tb0 = 0.0;
    for (int round = 0; round < 7; round++) {
        double dt, ta, tb;
        double g = bench_once(ldc, K, reps, &dt, &ta, &tb);
        if (round == 0) { dt0 = dt; ta0 = ta; tb0 = tb; }
        if (g > best) best = g;
    }
    if (!isfinite(best) || best <= 0.0) {
        /* v2 的 0.00 之谜定位数据: 原始时钟读数全量打出 */
        any_invalid = 1;
        printf("  [%-28s] INVALID  DIAG: best=%g | round0 t0=%.9f t1=%.9f dt=%.9f s\n",
               name, best, ta0, tb0, dt0);
        return;
    }
    printf("  [%-28s] ldc=%4d K=%4d reps=%5d  %8.2f GFLOPS   (round0 dt=%8.3f ms)\n",
           name, ldc, K, reps, best, dt0 * 1e3);
}

int main(void) {
    printf("=== microkernel-only benchmark v3 (diag) ===\n");

    /* A/B 按 K 上限 1024 分配 (v1 的 32x256 在 K=512 用例越界读) */
    A32 = malloc(sizeof(float) * 32 * 1024);
    B32 = malloc(sizeof(float) * 1024 * 32);
    C32 = malloc(sizeof(float) * 1024 * 1024);
    for (int i = 0; i < 32 * 1024; i++) A32[i] = 0.01f * ((i % 7) - 3);
    for (int i = 0; i < 1024 * 32; i++) B32[i] = 0.01f * ((i % 5) - 2);
    memset(C32, 0, sizeof(float) * 1024 * 1024);

    /* 探针接线自检: probe(ldc=32, K=64) vs 纯 C 参照 — 证明测的不是空转
     * A col-major 32x64: A[i][k] = A32[k*32+i]; B row-major 64x32: B[k][j] = B32[k*32+j] */
    {
        float *ref = malloc(sizeof(float) * 32 * 32);
        memset(ref, 0, sizeof(float) * 32 * 32);
        memset(C32, 0, sizeof(float) * 32 * 32);
        for (int k = 0; k < 64; k++)
            for (int i = 0; i < 32; i++) {
                float a = A32[k * 32 + i];
                for (int j = 0; j < 32; j++)
                    ref[i * 32 + j] += a * B32[k * 32 + j];
            }
        kirby_kernel_probe(C32, 32, A32, B32, 64);
        int bad = 0;
        for (int i = 0; i < 32 * 32; i++)
            if (fabsf(C32[i] - ref[i]) > 1e-4f + 1e-3f * fabsf(ref[i])) bad++;
        printf("probe wiring check (ldc=32, K=64): %s (%d/1024 mismatch)\n\n",
               bad ? "FAIL" : "OK", bad);
        free(ref);
        memset(C32, 0, sizeof(float) * 1024 * 1024);
    }

    /* (a) ldc=32: 连续 32x32 块, 等价打包版布局 (直连收益归因基准点) */
    bench_case("packed-layout (ldc=32)", 32, 128, 2000);
    /* (b) ldc=1024: 直连 1024 大矩阵的行距 (离散写回) */
    bench_case("direct-C (ldc=1024)", 1024, 128, 2000);
    /* (c) ldc=512/256/128: 中间档位 (bench.c 实际扫描的 N 序列) */
    bench_case("direct-C (ldc=512)", 512, 128, 2000);
    bench_case("direct-C (ldc=256)", 256, 128, 2000);
    bench_case("direct-C (ldc=128)", 128, 128, 2000);
    /* (d) ldc=32 K=512: K 变长吞吐 (更长 K 循环摊薄载入写回) */
    bench_case("packed-layout (ldc=32) K=512", 32, 512, 500);
    /* (e) ldc=1024 K=512: 直连大 ldc 下 K 变长 */
    bench_case("direct-C (ldc=1024) K=512", 1024, 512, 500);

    if (any_invalid) {
        printf("\nRESULT: 有无效用例 — 见上方 DIAG 行 (v2 的 0.00 之谜定位数据)\n");
        free(A32); free(B32); free(C32);
        return 3;
    }
    printf("\n(对比 bench.c 整体数字: 整体 GFLOPS / kernel GFLOPS = 调度效率)\n");
    printf("(M4 真实 SME 峰值 ~ kernel GFLOPS 上限; 以此修正 bench.c 口径)\n");
    free(A32); free(B32); free(C32);
    return 0;
}
