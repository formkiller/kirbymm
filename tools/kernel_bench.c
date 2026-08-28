/*
 * kernel_bench.c v6 — 差分计时 (differential chronometry)
 *
 * v5 dump 破案 (M4):
 *   - 12 个原始 dt ≈ 7643054.68s 恒定 + 每轮递增 0.9~4.1ms → 7643054.68s ≈ 88.5 天 = 开机时长
 *   - 即: t1 读数 (2000 次 kernel 调用后) = 真实绝对时间; t0 读数 (预热后) 冻结在开机零点
 *   - 相邻 dt 差分 = 真实每轮耗时 (恒定偏移相消); 递减趋势 = 热身, 稳态 ~0.9-1.1ms/2000 次
 *   - 交叉验证: bench.c 779 GFLOPS → ~336ns/kernel 调用; v5 稳态 ldc=32 → ~460ns/次 (~570 GFLOPS)
 *     → packed 布局病态假设初步成立, 待 v6 差分对照 ldc=512/1024 确认
 * v6 协议 — 差分计时, 不解释任何单次读数的绝对值:
 *   1. timed_round 原样保留 (作为"时间戳发生器": 每轮末尾的 t1 读数已被证实真实)
 *   2. 每用例 42 轮收 42 个 dt, 差分 D(i)=dt(i+1)-dt(i) (41 个), 窗口 (0,10)s 过滤
 *   3. 三口径: min(D) 峰值定标 / median(D) 稳健 / tail-mean(D) 时间序末 10 个 (热身后稳态)
 *   4. 端到端校验: 差分测已知 50ms nanosleep, 须落在 [45,60]ms
 *   5. 冻结点定位: case (a) 前 5 轮记录 预热前/预热后 各一次原始读数 — 定位 SME 往返是否即冻结点
 *   6. 全部保留: PRE/POST 自检, IEEE 探针, 接线自检, 编译戳, 防消除 sink
 *
 * 方法: 预分配 A/B/C, kernel 重复调用 (C += A*B 累加语义)
 * 数值安全: K=512, reps=500 时 C 元素 ~ N(0, 20^2), 远离溢出/denormal
 * v2 修复保留: A32/B32 按 K 上限 1024 分配
 */
#ifndef __APPLE__
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <setjmp.h>

#ifdef __APPLE__
  #include <mach/mach_time.h>
#endif

/* ---------- 计时器候选层 ---------- */

static double read_clock(void);   /* 前向声明 (read_mach 非 APPLE 回退用) */

static double read_mach(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t tb;
    static int inited = 0;
    if (!inited) { mach_timebase_info(&tb); inited = 1; }
    return (double)mach_absolute_time() * (double)tb.numer / (double)tb.denom / 1e9;
#else
    return read_clock();
#endif
}

static double read_clock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#if defined(__aarch64__)
static double read_cntvct(void) {
    static double inv_freq = 0.0;
    if (inv_freq == 0.0) {
        unsigned long long f;
        __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
        inv_freq = (double)f;
    }
    unsigned long long v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return (double)v / inv_freq;
}

#ifdef __APPLE__
/* SIGILL 探测 cntvct_el0 用户态可读性 */
static sigjmp_buf g_ill_jb;
static void on_sigill(int sig) { (void)sig; siglongjmp(g_ill_jb, 1); }
static int cntvct_available(void) {
    struct sigaction sa, old;
    sa.sa_handler = on_sigill;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGILL, &sa, &old);
    if (sigsetjmp(g_ill_jb, 1) == 0) {
        unsigned long long f, v;
        __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
        __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
        sigaction(SIGILL, &old, NULL);
        return (f > 0);
    }
    sigaction(SIGILL, &old, NULL);
    return 0;
}
#else
static int cntvct_available(void) { return 1; }
#endif   /* __APPLE__ */

#else    /* 非 aarch64 宿主: 桩 */
static double read_cntvct(void) { return 0.0; }
static int cntvct_available(void) { return 0; }
#endif   /* __aarch64__ */

typedef struct { const char *name; double (*read)(void); int avail; } timer_t;
static timer_t g_timers[3] = {
    { "mach_absolute_time", read_mach,   0 },
    { "clock_gettime",      read_clock,  0 },
    { "cntvct_el0",         read_cntvct, 0 },
};
static const int N_TIMERS = 3;

/* 自检: 预热 3 读 + 50ms sleep + 前后读; 返回 dt */
static double timer_selftest_dt(int id) {
    double sink;
    sink = g_timers[id].read(); (void)sink;
    sink = g_timers[id].read(); (void)sink;
    double t0 = g_timers[id].read();
    struct timespec req = {0, 50 * 1000 * 1000};
    nanosleep(&req, NULL);
    double t1 = g_timers[id].read();
    return t1 - t0;
}

static void run_selftests(const char *phase) {
    printf("[%s SME] timer selftests (50ms sleep):\n", phase);
    for (int i = 0; i < N_TIMERS; i++) {
        if (!g_timers[i].avail) { printf("  timer[%s] unavailable\n", g_timers[i].name); continue; }
        double d = timer_selftest_dt(i);
        int ok = (d > 0.03 && d < 1.0);
        printf("  timer[%s] -> %.6f s  [%s]\n", g_timers[i].name, d, ok ? "OK" : "DEAD");
    }
}

/* ---------- 被测对象 ---------- */
#include "kirby.h"

void kirby_kernel_probe(float *C, int ldc, const float *A, const float *B, int K);

static float *A32, *B32, *C32;
static double benchmark_sink = 0.0;

/* 冻结点定位用: 预热前/后各一次原始读数 (仅 debug 轮记录) */
static double g_dbg_pre = 0.0, g_dbg_post = 0.0;
static int g_dbg_on = 0;

/* 时间戳发生器: 一轮 = 预热 + 读 t0(3取max) + reps 次 kernel + 读 t1(3取max)
 * 返回 dt = t1 - t0 (含未知恒定偏移 C, 差分后相消) */
static double timed_round(int tid, int ldc, int K, int reps) {
    if (g_dbg_on) g_dbg_pre = g_timers[tid].read();
    kirby_kernel_probe(C32, ldc, A32, B32, K);               /* 预热 (SME) */
    if (g_dbg_on) g_dbg_post = g_timers[tid].read();
    volatile double sink = g_timers[tid].read(); (void)sink; /* 吸收坏读 */
    sink = g_timers[tid].read(); (void)sink;
    double t0 = g_timers[tid].read();                        /* t0: 3 读取 max */
    for (int i = 0; i < 2; i++) {
        double t = g_timers[tid].read();
        if (t > t0) t0 = t;
    }
    for (int r = 0; r < reps; r++)
        kirby_kernel_probe(C32, ldc, A32, B32, K);
    double t1 = g_timers[tid].read();                        /* t1: 3 读取 max (v5 证实此位真实) */
    for (int i = 0; i < 2; i++) {
        double t = g_timers[tid].read();
        if (t > t1) t1 = t;
    }
    benchmark_sink += C32[0];
    return t1 - t0;
}

/* 差分通道端到端校验: 已知 50ms nanosleep, 差分读数须落在 [45,60]ms */
static int differential_validation(int tid) {
    for (int r = 0; r < 2000; r++)                           /* ~1ms 热身列车, 置于 v5 证实的"真实位" */
        kirby_kernel_probe(C32, 32, A32, B32, 128);
    double ta = g_timers[tid].read();
    struct timespec req = {0, 50 * 1000 * 1000};
    nanosleep(&req, NULL);
    double tb = g_timers[tid].read();
    double d = tb - ta;
    int ok = (d > 0.045 && d < 0.060);
    printf("differential check (50ms sleep): %.6f s  [%s]\n\n", d, ok ? "OK" : "FAIL");
    return ok;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* 差分统计: 42 轮 dt → 41 个差分 D; min / median / tail-mean(时间序末 10 个)
 * GFLOPS = 每轮 flops / D (D 即每轮真实耗时) */
static void bench_case(const char *name, int ldc, int K, int reps,
                       int tid, int dump_raw) {
    double dts[42];
    int n = 0;
    for (int attempt = 0; attempt < 42; attempt++) {
        g_dbg_on = dump_raw && attempt < 5;
        double dt = timed_round(tid, ldc, K, reps);
        g_dbg_on = 0;
        dts[n++] = dt;
        if (dump_raw && attempt < 20)
            printf("    try#%-2d raw dt=%.9f s%s\n", attempt + 1, dt,
                   (attempt < 5) ? "   [preheat-reads below]" : "");
        if (dump_raw && attempt < 5)
            printf("      pre-preheat=%.9f  post-preheat=%.9f  (freeze localization)\n",
                   g_dbg_pre, g_dbg_post);
    }

    double deltas[41];
    int nd = 0, n_rej = 0;
    for (int i = 0; i + 1 < n; i++) {
        double d = dts[i + 1] - dts[i];
        if (d > 0.0 && d < 10.0) deltas[nd++] = d;
        else n_rej++;
    }
    if (nd == 0) {
        printf("  [%-28s] NO VALID DELTAS (channel unstable)\n", name);
        return;
    }
    /* tail-mean: 时间序末 10 个差分的均值 (热身后的稳态口径) — 排序前先算 */
    int tcount = (nd < 10) ? nd : 10, tstart = nd - tcount;
    double tsum = 0.0;
    for (int i = tstart; i < nd; i++) tsum += deltas[i];
    double tail_mean = tsum / tcount;
    /* min / median: 排序副本 */
    double sorted[41];
    memcpy(sorted, deltas, sizeof(double) * (size_t)nd);
    qsort(sorted, (size_t)nd, sizeof(double), cmp_double);
    double dmin = sorted[0], dmed = sorted[nd / 2];

    double flops = 2.0 * 32.0 * 32.0 * K * reps;
    printf("  [%-28s] ldc=%4d K=%4d\n"
           "      GFLOPS: minD=%7.2f  medianD=%7.2f  tailD=%7.2f"
           "   (deltas %d ok /%d rej; minD=%.3f ms medD=%.3f ms tailD=%.3f ms)\n",
           name, ldc, K,
           flops / dmin / 1e9, flops / dmed / 1e9, flops / tail_mean / 1e9,
           nd, n_rej, dmin * 1e3, dmed * 1e3, tail_mean * 1e3);
}

int main(void) {
    printf("=== microkernel-only benchmark v6 (differential) ===\n");
    printf("build stamp: %s %s\n", __DATE__, __TIME__);

    {   /* IEEE 除零探针 */
        volatile double one = 1.0, zero = 0.0;
        double r = one / zero;
        printf("IEEE check: 1/0 -> %g (expect inf)%s\n", r,
               (isinf(r) && r > 0) ? "" : "  << ANOMALY!");
    }

    /* 计时器可用性 + PRE/POST 自检 */
    g_timers[2].avail = cntvct_available();
    g_timers[0].avail = 1; g_timers[1].avail = 1;
    run_selftests("PRE");

    /* A/B 按 K 上限 1024 分配 */
    A32 = malloc(sizeof(float) * 32 * 1024);
    B32 = malloc(sizeof(float) * 1024 * 32);
    C32 = malloc(sizeof(float) * 1024 * 1024);
    for (int i = 0; i < 32 * 1024; i++) A32[i] = 0.01f * ((i % 7) - 3);
    for (int i = 0; i < 1024 * 32; i++) B32[i] = 0.01f * ((i % 5) - 2);
    memset(C32, 0, sizeof(float) * 1024 * 1024);

    /* 探针接线自检 */
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

    run_selftests("POST");

    /* 选主计时器: POST 自检通过的第一个 */
    int tid = -1;
    for (int i = 0; i < N_TIMERS; i++) {
        if (!g_timers[i].avail) continue;
        double d = timer_selftest_dt(i);
        if (d > 0.03 && d < 1.0) { tid = i; break; }
    }
    if (tid < 0) {
        printf("FATAL: 所有计时器 POST 自检失败\n");
        return 2;
    }
    printf("  bench timer: %s\n", g_timers[tid].name);
    if (!differential_validation(tid)) {
        printf("FATAL: 差分通道未通过 50ms 校验 — 差分也不可信, 停止测量\n");
        free(A32); free(B32); free(C32);
        return 5;
    }

    /* (a) 基准点 + 原始读数倾泻 + 冻结点定位 */
    bench_case("packed-layout (ldc=32)", 32, 128, 2000, tid, 1);
    /* (b) 直连大矩阵行距 */
    bench_case("direct-C (ldc=1024)", 1024, 128, 2000, tid, 0);
    /* (c) 中间档位 (bench.c 实际 N 序列; 779 来自 N=512) */
    bench_case("direct-C (ldc=512)", 512, 128, 2000, tid, 0);
    bench_case("direct-C (ldc=256)", 256, 128, 2000, tid, 0);
    bench_case("direct-C (ldc=128)", 128, 128, 2000, tid, 0);
    /* (d) K 变长 */
    bench_case("packed-layout (ldc=32) K=512", 32, 512, 500, tid, 0);
    bench_case("direct-C (ldc=1024) K=512", 1024, 512, 500, tid, 0);

    printf("\nsink=%.6g (防消除校验)\n", benchmark_sink);
    printf("(归因判读: minD 口径 ldc=512/1024 vs ldc=32 — 差距大则 packed 布局病态实锤)\n");
    printf("(779 GFLOPS 全 GEMM 折算 ~336ns/调用; 对照此处每调用 ns 定位调度开销)\n");
    free(A32); free(B32); free(C32);
    return 0;
}
