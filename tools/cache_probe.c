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
 * 注意: -O3 向量化会把顺序访问合并, 测的是吞吐而非单次延迟
 * 此法不可靠, 仅作参考. 真实延迟应用 pointer chasing (后续补) */
static void probe_stride_latency(void) {
    printf("--- stride latency probe (受 -O3 向量化影响, 数字仅供参考) ---\n");
    printf("%-12s %12s\n", "workset", "throughput(ns/elem)");
    printf("-----------------------------\n");

    size_t sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384};
    int n = (int)(sizeof(sizes)/sizeof(sizes[0]));

    for (int si = 0; si < n; si++) {
        size_t bytes = sizes[si] * 1024;
        size_t count = bytes / sizeof(uint64_t);
        uint64_t *buf = (uint64_t *)malloc(bytes);
        if (!buf) continue;
        memset(buf, 0, bytes);

        volatile uint64_t sink = 0;
        double t0 = now_sec();
        int reps = 1000;
        for (int r = 0; r < reps; r++) {
            for (size_t i = 0; i < count; i++) {
                buf[i] += i;
                sink += buf[i];
            }
        }
        double dt = now_sec() - t0;
        double ns_per = dt / reps / count * 1e9;

        printf("%-8zu KB %12.2f\n", sizes[si], ns_per);
        free(buf);
    }
    printf("\n(注: 真实 L1/L2 边界应以 sysctl 容量为准, 此探测仅供交叉参考)\n\n");
}

/* 带宽探测: memcpy 测 B_max (不能被优化掉, OS 高度优化)
 * memcpy = 1 read + 1 write, 纯读带宽 ≈ 总带宽的一半 */
static void probe_bandwidth(void) {
    printf("--- memory bandwidth probe (B_max for Eq.5) ---\n");
    size_t bytes = 64 * 1024 * 1024;  /* 64MB, 远超 L2, 测真实内存带宽 */
    uint8_t *src = (uint8_t *)malloc(bytes);
    uint8_t *dst = (uint8_t *)malloc(bytes);
    if (!src || !dst) { printf("alloc failed\n"); return; }
    memset(src, 0xAB, bytes);

    /* warmup */
    memcpy(dst, src, bytes);
    __asm__ volatile("" ::: "memory");  /* 阻止跨迭代消除 */
    int reps = 50;
    double t0 = now_sec();
    for (int r = 0; r < reps; r++) {
        memcpy(dst, src, bytes);
        __asm__ volatile("" ::: "memory");  /* 关键: 阻止 -O3 把多次 memcpy 合并/消除 */
    }
    double dt = now_sec() - t0;
    /* memcpy = read + write, 总数据移动 = 2*bytes*reps */
    double total_gbs = (double)bytes * 2.0 * reps / dt / 1e9;
    double read_gbs = (double)bytes * reps / dt / 1e9;  /* 纯读估计 */

    printf("memcpy 64MB x%d: total %.2f GB/s (read+write)\n", reps, total_gbs);
    printf("B_max (read est) ≈ %.2f GB/s\n\n", read_gbs);
    printf("代入 Eq.5: I_max = P_peak / B_max = 512 / %.2f = %.2f FLOP/byte\n",
           read_gbs, 512.0 / read_gbs);
    free(src); free(dst);
}

/* 相联度假设 (Apple 不暴露, 用典型值, 待 conflict 探测验证) */
static void print_assoc_assumption(void) {
    printf("--- associativity assumption (待验证) ---\n");
    printf("Apple M4 P-core 典型: L1d 8-way, L2 16-way (公开资料不确认)\n");
    printf("L1: 64KB / (8 × 128B) = 64 组\n");
    printf("L2: 4MB / (16 × 128B) = 2048 组\n");
    printf("(后续可加 conflict miss 探测实测相联度)\n\n");
}

int main(void) {
    printf("=== M4 cache geometry probe ===\n\n");
    probe_sysctl();
    probe_stride_latency();
    probe_bandwidth();
    print_assoc_assumption();
    printf("\n=== probe done ===\n");
    return 0;
}
