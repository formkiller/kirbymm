/*
 * kernel_bench.c v4 — 计时器自验证版
 *
 * v3 在 M4 上的诊断结论:
 *   - t0=t1=0.000000000 恒成立 → clock_gettime(CLOCK_MONOTONIC) 在该环境恒返回 {0,0}
 *   - bench.c 的 mach_absolute_time 路径此前打印过 779 GFLOPS → mach 计时器已证实可用
 *   - v3 源码下 dt=0 应得 g=+inf → DIAG "best=0" 在 IEEE 语义下不可能出现
 *     → 两种可能: (a) 该环境除零非 IEEE (返回 0/NaN); (b) 二进制与仓库源码分歧
 * v4 对策:
 *   1. APPLE 首选 mach_absolute_time (已证实可用), clock_gettime 作候选;
 *      每个候选过 50ms sleep 自检 (读数差必须落在 (0.03, 1.0) s), 死了换下一个
 *   2. IEEE 除零探针: volatile 1.0/0.0 必须 = +inf (运行期除法, 防 -O3 折叠)
 *   3. 哨兵算术: dt<=0 → 返回 -1 (永不产生 inf/NaN), 全灭轮打印 TIMER DEAD + 原始读数
 *   4. __DATE__/__TIME__ 编译戳 → 甄别陈旧二进制 (与 cmake 构建时间对照)
 *   5. 探针接线自检保留 (v3 已验证 probe 数值正确: 0/1024 mismatch)
 *
 * 方法: 预分配 A/B/C, kernel 重复调用 (C += A*B 累加语义), best-of-7 取最快轮
 * 数值安全: K=512, reps=500 时 C 元素 ~ N(0, 20^2), 远离溢出/denormal
 * v2 修复保留: A32/B32 按 K 上限 1024 分配 (v1 在 K=512 用例越界读堆)
 */
#ifndef __APPLE__
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
  #include <mach/mach_time.h>
#endif

/* ---------- 计时器层: 双候选 + 运行时自检 ---------- */
static int g_timer = 0;   /* 1 = mach_absolute_time (APPLE), 2 = clock_gettime */

#ifdef __APPLE__
static double now_mach(void) {
    static mach_timebase_info_data_t tb;
    static int inited = 0;
    if (!inited) { mach_timebase_info(&tb); inited = 1; }
    return (double)mach_absolute_time() * (double)tb.numer / (double)tb.denom / 1e9;
}
#endif

static double now_clock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double now_sec(void) {
#ifdef __APPLE__
    if (g_timer == 1) return now_mach();
#endif
    return now_clock();
}

static const char *timer_name(int id) {
#ifdef __APPLE__
    if (id == 1) return "mach_absolute_time";
#else
    (void)id;
#endif
    return "clock_gettime(CLOCK_MONOTONIC)";
}

/* 自检: 50ms sleep 前后读数差必须合理 (NaN/0/过大都判死); 返回 1=可用 */
static int timer_selftest(int id) {
    g_timer = id;
    double t0 = now_sec();
    struct timespec req = {0, 50 * 1000 * 1000};   /* 50 ms */
    nanosleep(&req, NULL);
    double t1 = now_sec();
    double d = t1 - t0;
    int ok = (d > 0.03 && d < 1.0);
    printf("  timer[%s] selftest: 50ms sleep -> %.6f s  [%s]\n",
           timer_name(id), d, ok ? "OK" : "DEAD");
    return ok;
}

/* ---------- 被测对象 ---------- */
#include "kirby.h"

/* [仅测试用] 探针: primary.c 导出, 直通 micro_32x32_kupdate */
void kirby_kernel_probe(float *C, int ldc, const float *A, const float *B, int K);

static float *A32, *B32, *C32;

/* 单轮计时: kernel 重复调用 reps 次; 回传 t0/t1/dt 供诊断
 * 哨兵: dt<=0 返回 -1 (计时死/倒退), 永不产生 inf/NaN */
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
    if (!(dt > 0.0)) return -1.0;
    return 2.0 * 32.0 * 32.0 * K * reps / dt / 1e9;
}

static int any_invalid = 0;

/* best-of-7: 峰值定标取最快轮 (OS 抖动/降频轮直接淘汰) */
static void bench_case(const char *name, int ldc, int K, int reps) {
    double best = -1.0, dt0 = -1.0, ta0 = 0.0, tb0 = 0.0;
    for (int round = 0; round < 7; round++) {
        double dt, ta, tb;
        double g = bench_once(ldc, K, reps, &dt, &ta, &tb);
        if (round == 0) { dt0 = dt; ta0 = ta; tb0 = tb; }
        if (g > best) best = g;
    }
    if (best < 0.0) {
        any_invalid = 1;
        printf("  [%-28s] TIMER DEAD | round0 t0=%.9f t1=%.9f dt=%.9f (all 7 rounds)\n",
               name, ta0, tb0, dt0);
        return;
    }
    printf("  [%-28s] ldc=%4d K=%4d reps=%5d  %8.2f GFLOPS   (round0 dt=%8.3f ms)\n",
           name, ldc, K, reps, best, dt0 * 1e3);
}

int main(void) {
    printf("=== microkernel-only benchmark v4 (selftest) ===\n");
    printf("build stamp: %s %s\n", __DATE__, __TIME__);

    /* IEEE 除零探针 (volatile 防 -O3 折叠): 非零/0 必须 = +inf */
    {
        volatile double one = 1.0, zero = 0.0;
        double r = one / zero;
        int ieee_ok = isinf(r) && r > 0;
        printf("IEEE check: 1/0 -> %g (expect inf)%s\n",
               r, ieee_ok ? "" : "  << ANOMALY: 除零非IEEE!");
    }

    /* 计时器自检 + 选择 (APPLE: mach 优先 — bench.c 779 已证实可用) */
    int picked = 0;
#ifdef __APPLE__
    if (!picked && timer_selftest(1)) picked = 1;
#endif
    if (!picked && timer_selftest(2)) picked = 2;
    if (!picked) {
        printf("FATAL: 所有计时器自检失败 (系统时钟冻结?) — 无法测量\n");
        return 2;
    }
    printf("  using timer: %s\n\n", timer_name(g_timer));

    /* A/B 按 K 上限 1024 分配 (v1 的 32x256 在 K=512 用例越界读) */
    A32 = malloc(sizeof(float) * 32 * 1024);
    B32 = malloc(sizeof(float) * 1024 * 32);
    C32 = malloc(sizeof(float) * 1024 * 1024);
    for (int i = 0; i < 32 * 1024; i++) A32[i] = 0.01f * ((i % 7) - 3);
    for (int i = 0; i < 1024 * 32; i++) B32[i] = 0.01f * ((i % 5) - 2);
    memset(C32, 0, sizeof(float) * 1024 * 1024);

    /* 探针接线自检: probe(ldc=32, K=64) vs 纯 C 参照 — 证明测的不是空转 */
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
        printf("\nRESULT: 有无效用例 — 见上方 TIMER DEAD/ANOMALY 行\n");
        free(A32); free(B32); free(C32);
        return 3;
    }
    printf("\n(对比 bench.c 整体数字: 整体 GFLOPS / kernel GFLOPS = 调度效率)\n");
    printf("(M4 真实 SME 峰值 ~ kernel GFLOPS 上限; 以此修正 bench.c 口径)\n");
    free(A32); free(B32); free(C32);
    return 0;
}
