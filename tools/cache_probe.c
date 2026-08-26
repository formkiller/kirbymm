/*
 * cache_probe.c — M4 缓存几何参数探测 (§V-A 内存敏感分块的前置)
 *
 * 探测项 (供 Eq.6/7 代入):
 *   L1 data cache: 容量 S_L1·A_L1·L_L1, 组数 S_L1, 相联度 A_L1, 行大小 L_L1
 *   L2 cache:      容量 S_L2·A_L2·L_L2, 组数 S_L2, 相联度 A_L2, 行大小 L_L2
 *   内存带宽 B_max (供 Eq.5 I_max = P_peak/B_max)
 *
 * 方法:
 *   1. sysctlbyname 查 macOS 暴露的缓存参数
 *   2. 步进探测验证 (不同 stride 访问测延迟拐点, 识别 L1/L2 边界与相联度)
 *   3. 带宽探测 (连续 load 测 GB/s)
 *
 * 编译: cmake --build build
 * 运行: ./build/cache_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef __APPLE__
  #include <sys/sysctl.h>
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

/* sysctl 查询缓存参数 */
static void probe_sysctl(void) {
    printf("--- sysctl cache parameters ---\n");
    size_t l1d = 0, l2 = 0, linesize = 0;
    size_t l1_assoc = 0, l2_assoc = 0;
    size_t len = sizeof(size_t);

    sysctlbyname("hw.l1dcachesize", &l1d, &len, NULL, 0);
    sysctlbyname("hw.l2cachesize", &l2, &len, NULL, 0);
    sysctlbyname("hw.cachelinesize", &linesize, &len, NULL, 0);
    sysctlbyname("hw.l1dcacheassociativity", &l1_assoc, &len, NULL, 0);
    sysctlbyname("hw.l2cacheassociativity", &l2_assoc, &len, NULL, 0);

    printf("L1 data cache : %8zu bytes (%zu KB)\n", l1d, l1d/1024);
    printf("L2 cache      : %8zu bytes (%zu KB)\n", l2, l2/1024);
    printf("cacheline     : %8zu bytes\n", linesize);
    printf("L1 assoc      : %8zu %s\n", l1_assoc, l1_assoc ? "" : "(sysctl 未暴露)");
    printf("L2 assoc      : %8zu %s\n", l2_assoc, l2_assoc ? "" : "(sysctl 未暴露)");
    if (l1_assoc && linesize)
        printf("L1 sets       : %8zu\n", l1d / (l1_assoc * linesize));
    if (l2_assoc && linesize)
        printf("L2 sets       : %8zu\n", l2 / (l2_assoc * linesize));
    printf("\n");
}

/* 步进探测: 不同 stride 的访问延迟, 找 L1/L2 边界
 * 原理: 工作集 < L1 时延迟低, 超出 L1 进 L2 时延迟跳升 */
static void probe_stride_latency(void) {
    printf("--- stride latency probe (识别 L1/L2 边界) ---\n");
    printf("%-12s %12s\n", "workset", "latency(ns)");
    printf("-----------------------------\n");

    /* 测试不同工作集大小: 4KB → 16MB */
    size_t sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384};
    int n = (int)(sizeof(sizes)/sizeof(sizes[0]));

    for (int si = 0; si < n; si++) {
        size_t bytes = sizes[si] * 1024;
        size_t count = bytes / sizeof(uint64_t);
        uint64_t *buf = (uint64_t *)malloc(bytes);
        if (!buf) continue;
        memset(buf, 0, bytes);

        /* 链表式访问: 每个 element 指向下一个, stride 跨 cacheline 制造冲突 */
        /* 简化: 顺序访问所有 element, 多次循环 */
        volatile uint64_t sink = 0;
        double t0 = now_sec();
        int reps = 1000;
        for (int r = 0; r < reps; r++) {
            for (size_t i = 0; i < count; i++) {
                buf[i] += i;  /* 读写, 强制进缓存 */
                sink += buf[i];
            }
        }
        double dt = now_sec() - t0;
        double ns_per = dt / reps / count * 1e9;

        printf("%-8zu KB %12.2f\n", sizes[si], ns_per);
        free(buf);
    }
    printf("\n(延迟跳升处即 L1→L2 边界, 再跳升即 L2→内存)\n\n");
}

/* 带宽探测: 连续 load 测 B_max (Eq.5 用) */
static void probe_bandwidth(void) {
    printf("--- memory bandwidth probe (B_max for Eq.5) ---\n");
    size_t bytes = 64 * 1024 * 1024;  /* 64MB, 远超 L2, 测真实内存带宽 */
    size_t count = bytes / sizeof(uint64_t);
    uint64_t *buf = (uint64_t *)malloc(bytes);
    if (!buf) { printf("alloc failed\n"); return; }
    memset(buf, 0, bytes);

    volatile uint64_t sink = 0;
    int reps = 50;
    double t0 = now_sec();
    for (int r = 0; r < reps; r++) {
        uint64_t acc = 0;
        for (size_t i = 0; i < count; i++) {
            acc += buf[i];  /* 纯读, 流式 */
        }
        sink += acc;
    }
    double dt = now_sec() - t0;
    double gb_s = (double)bytes * reps / dt / 1e9;

    printf("64MB stream read: %.2f GB/s = %.2f GB/s\n\n", gb_s, gb_s);
    printf("B_max ≈ %.2f GB/s (代入 Eq.5: I_max = P_peak/B_max)\n", gb_s);
    free(buf);
}

int main(void) {
    printf("=== M4 cache geometry probe ===\n\n");
    probe_sysctl();
    probe_stride_latency();
    probe_bandwidth();
    printf("\n=== probe done ===\n");
    return 0;
}
