/*
 * kernel_bench.c v5 — 穿越式计时 (garbage-immune rounds)
 *
 * v2~v4 诊断结论链 (M4):
 *   - IEEE 除零正常 (v4: 1/0=inf), 探针接线正常 (0/1024), mach 计时器 SME 前正常
 *   - 故障模式: 计时读数在 SME 密集调用上下文中错乱 —
 *     v3 clock_gettime 读数恒 0; v4 mach 在 bench 中 dt=0/跳变, 在 selftest 中正常
 *   - 推论: 不是计时器坏, 是 SME 往返 (smstart/smstop) 之后的读数窗口不可靠
 *   - bench.c 的 779 仍可信: t0(SME前)/t1(SME后) 均真实, 否则会打印负数或 0.00
 * v5 对策 — 不修计时器, 让每轮计时自带防伪:
 *   1. 三计时器候选: mach_absolute_time / clock_gettime / cntvct_el0 (raw 计数器,
 *      与前两者 API 无关的独立见证; SIGILL 探测可用性)
 *   2. 每轮: t0 取 3 读 max (吸收 0 值垃圾), t1 取 3 读 max;
 *      校验 1e-6 s < dt < 10 s (round 真实耗时 ~0.7-1.5 ms, 上下界放宽到 6 个数量级)
 *      — 0 读数 / 冻结 / 巨大跳变全部进不了统计
 *   3. 每用例最多 42 次尝试收满 7 个有效轮, best-of-valid; 打印 n_valid
 *   4. 用例 (a) 倾泻前 12 次尝试的原始读数 (t0/t1/dt/verdict) — 垃圾模式留档
 *   5. SME 前/后各跑一轮计时器自检 (50ms sleep) — 直接量化 "SME 后计时器失灵"
 *   6. 防优化消除: 每轮后 benchmark_sink += C32[0], 结束打印 (调用链必须保留)
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
/* SIGILL 探测 cntvct_el0 用户态可读性 (macOS 一般放行, 防意外) */
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
/* Linux aarch64 用户态恒可读 generic timer */
static int cntvct_available(void) { return 1; }
#endif   /* __APPLE__ */

#else    /* 非 aarch64 宿主: cntvct 桩 (标记不可用) */
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

/* 自检: 预热 3 读 (吸收坏读) + 50ms sleep + 前后读; 返回 dt, 调用方判界 */
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

/* 挑一个 SME 后自检通过的计时器; 全灭返回 -1 */
static int pick_timer_post_sme(void) {
    for (int i = 0; i < N_TIMERS; i++) {
        if (!g_timers[i].avail) continue;
        double d = timer_selftest_dt(i);
        if (d > 0.03 && d < 1.0) return i;
    }
    return -1;
}

/* ---------- 被测对象 ---------- */
#include "kirby.h"

void kirby_kernel_probe(float *C, int ldc, const float *A, const float *B, int K);

static float *A32, *B32, *C32;
static double benchmark_sink = 0.0;   /* 防优化消除 */

/* 单轮: 预热 + 吸收读 + t0(3读max) + reps 次 kernel + t1(3读max); 返回 dt (原始) */
static double timed_round(int tid, int ldc, int K, int reps) {
    kirby_kernel_probe(C32, ldc, A32, B32, K);              /* 预热 (SME) */
    volatile double sink = g_timers[tid].read(); (void)sink; /* 吸收坏读 */
    sink = g_timers[tid].read(); (void)sink;
    double t0 = g_timers[tid].read();
    for (int i = 0; i < 2; i++) {                            /* t0: 3 读取 max */
        double t = g_timers[tid].read();
        if (t > t0) t0 = t;
    }
    for (int r = 0; r < reps; r++)
        kirby_kernel_probe(C32, ldc, A32, B32, K);
    double t1 = g_timers[tid].read();                        /* t1: 3 读取 max */
    for (int i = 0; i < 2; i++) {
        double t = g_timers[tid].read();
        if (t > t1) t1 = t;
    }
    benchmark_sink += C32[0];                                /* 调用链活性 */
    return t1 - t0;
}

static int any_case_failed = 0;

/* best-of-valid: 最多 42 次尝试收 7 个有效轮 (1e-6 s < dt < 10 s) */
static void bench_case(const char *name, int ldc, int K, int reps,
                       int tid, int dump_raw) {
    double best = -1.0, dt0 = -1.0;
    int n_valid = 0, n_try = 0;
    for (int attempt = 0; attempt < 42 && n_valid < 7; attempt++) {
        double dt = timed_round(tid, ldc, K, reps);
        n_try++;
        if (dump_raw && attempt < 12)
            printf("    try#%-2d t=%-18s dt=%.9f s  [%s]\n",
                   attempt + 1, g_timers[tid].name, dt,
                   (dt > 1e-6 && dt < 10.0) ? "valid" : "REJECT");
        if (dt > 1e-6 && dt < 10.0) {
            if (dt0 < 0) dt0 = dt;
            n_valid++;
            double g = 2.0 * 32.0 * 32.0 * K * reps / dt / 1e9;
            if (g > best) best = g;
        }
    }
    if (n_valid == 0) {
        any_case_failed = 1;
        printf("  [%-28s] NO VALID ROUNDS (42 tries, timer=%s)\n",
               name, g_timers[tid].name);
        return;
    }
    printf("  [%-28s] ldc=%4d K=%4d reps=%5d  %8.2f GFLOPS   "
           "(valid %d/42, round0 dt=%8.3f ms)\n",
           name, ldc, K, reps, best, n_valid, dt0 * 1e3);
}

int main(void) {
    printf("=== microkernel-only benchmark v5 (garbage-immune) ===\n");
    printf("build stamp: %s %s\n", __DATE__, __TIME__);

    {   /* IEEE 除零探针 */
        volatile double one = 1.0, zero = 0.0;
        double r = one / zero;
        printf("IEEE check: 1/0 -> %g (expect inf)%s\n", r,
               (isinf(r) && r > 0) ? "" : "  << ANOMALY!");
    }

    /* 计时器可用性 + SME 前自检 */
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

    /* 探针接线自检 (SME!) */
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

    /* SME 后自检 — 直接量化 "SME 往返后计时器失灵" 假设 */
    run_selftests("POST");
    int tid = pick_timer_post_sme();
    if (tid < 0) {
        printf("\nFATAL: 所有计时器在 SME 后自检失败 — 20 次原始读数留档:\n");
        for (int i = 0; i < N_TIMERS; i++) {
            if (!g_timers[i].avail) continue;
            printf("  timer[%s]:", g_timers[i].name);
            for (int k = 0; k < 20; k++) {
                struct timespec req = {0, 1000 * 1000};  /* 1ms */
                nanosleep(&req, NULL);
                printf(" %.6f", g_timers[i].read());
            }
            printf("\n");
        }
        printf("-> 无计时通道; 改用 plan B (bench.c 全 GEMM 外推) 定标\n");
        free(A32); free(B32); free(C32);
        return 4;
    }
    printf("  bench timer: %s\n\n", g_timers[tid].name);

    /* (a) 基准点 + 原始读数倾泻 (垃圾模式留档) */
    bench_case("packed-layout (ldc=32)", 32, 128, 2000, tid, 1);
    /* (b) 直连大矩阵行距 */
    bench_case("direct-C (ldc=1024)", 1024, 128, 2000, tid, 0);
    /* (c) 中间档位 */
    bench_case("direct-C (ldc=512)", 512, 128, 2000, tid, 0);
    bench_case("direct-C (ldc=256)", 256, 128, 2000, tid, 0);
    bench_case("direct-C (ldc=128)", 128, 128, 2000, tid, 0);
    /* (d) K 变长 */
    bench_case("packed-layout (ldc=32) K=512", 32, 512, 500, tid, 0);
    bench_case("direct-C (ldc=1024) K=512", 1024, 512, 500, tid, 0);

    printf("\nsink=%.6g (防消除校验, 非零即调用链完整)\n", benchmark_sink);
    if (any_case_failed) {
        printf("RESULT: 有用例无有效轮 — 见上方, 计时环境需进一步留档\n");
        free(A32); free(B32); free(C32);
        return 3;
    }
    printf("(对比 bench.c 整体数字: 整体 GFLOPS / kernel GFLOPS = 调度效率)\n");
    printf("(M4 真实 SME 峰值 ~ kernel GFLOPS 上限; 以此修正 bench.c 口径)\n");
    free(A32); free(B32); free(C32);
    return 0;
}
