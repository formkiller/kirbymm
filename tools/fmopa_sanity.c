/*
 * fmopa_sanity.c — 四层诊断 streaming SVE/SME 链路 (v4)
 *
 * v4 修正:
 *   - test1 期望全 1 (fmov z0=#1.0), 之前打印标签错
 *   - 新增 test_ld1w: 从内存加载 [1..16] 到 z0, 绕开 index
 *   - test3 改用 ld1w 加载 z0/z1 (microkernel 实际方式), 不依赖 index
 *
 * 诊断分层:
 *   test1   minimal:  ptrue+fmov z0=#1.0+st1w          期望 [1,1,...,1]
 *   test2   index:    ptrue+index z0+st1w              期望 [1..16] (已知失败)
 *   test3   ld1w:     ptrue+ld1w z0+st1w               期望 [1..16] (绕开 index)
 *   test4   fmopa:    smstart za+ld1w z0/z1+fmopa+mova 期望 [1..16]
 */
#include <stdio.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>
#include <string.h>

static sigjmp_buf g_jmp;
static volatile sig_atomic_t g_sigill = 0;

static void on_sigill(int sig) {
    (void)sig;
    g_sigill = 1;
    siglongjmp(g_jmp, 1);
}

static struct sigaction install_handler(void) {
    struct sigaction sa, old;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_sigill;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGILL, &sa, &old);
    return old;
}

static void restore_handler(const struct sigaction *old) {
    sigaction(SIGILL, old, NULL);
}

/* test1: ptrue + fmov z0=#1.0 + st1w, 期望全 1 */
__attribute__((target("sve,sme")))
static int test_minimal(float *out) {
    struct sigaction old = install_handler();
    g_sigill = 0;
    int ok = 0;
    if (sigsetjmp(g_jmp, 1) == 0) {
        __asm__ volatile(
            "smstart sm\n\t"
            "ptrue p0.s, all\n\t"
            "fmov z0.s, #1.0\n\t"
            "st1w z0.s, p0, [%[out]]\n\t"
            "smstop sm\n\t"
            :
            : [out] "r" (out)
            : "memory"
        );
        ok = 1;
    }
    restore_handler(&old);
    return ok;
}

/* test2: ptrue + index z0 + st1w, 期望 1..16 (已知失败, 保留对照) */
__attribute__((target("sve,sme")))
static int test_index(float *out) {
    struct sigaction old = install_handler();
    g_sigill = 0;
    int ok = 0;
    if (sigsetjmp(g_jmp, 1) == 0) {
        __asm__ volatile(
            "smstart sm\n\t"
            "ptrue p0.s, all\n\t"
            "index z0.s, #1, #1\n\t"
            "st1w z0.s, p0, [%[out]]\n\t"
            "smstop sm\n\t"
            :
            : [out] "r" (out)
            : "memory"
        );
        ok = 1;
    }
    restore_handler(&old);
    return ok;
}

/* test3: ptrue + ld1w z0 + st1w, 期望 1..16 (绕开 index, 从内存加载) */
__attribute__((target("sve,sme")))
static int test_ld1w(const float *in, float *out) {
    struct sigaction old = install_handler();
    g_sigill = 0;
    int ok = 0;
    if (sigsetjmp(g_jmp, 1) == 0) {
        __asm__ volatile(
            "smstart sm\n\t"
            "ptrue p0.s, all\n\t"
            "ld1w z0.s, p0/z, [%[in]]\n\t"
            "st1w z0.s, p0, [%[out]]\n\t"
            "smstop sm\n\t"
            :
            : [in] "r" (in), [out] "r" (out)
            : "memory"
        );
        ok = 1;
    }
    restore_handler(&old);
    return ok;
}

/* test4: smstart za + ld1w z0/z1 + fmopa + mova + st1w, 期望 1..16 */
__attribute__((target("sve,sme")))
static int test_fmopa_ld1w(const float *a, const float *b, float *z0_out, float *za_out) {
    struct sigaction old = install_handler();
    g_sigill = 0;
    int ok = 0;
    if (sigsetjmp(g_jmp, 1) == 0) {
        __asm__ volatile(
            "smstart sm\n\t"
            "smstart za\n\t"
            "zero {za}\n\t"
            "ptrue p0.s, all\n\t"
            "ld1w z0.s, p0/z, [%[a]]\n\t"
            "ld1w z1.s, p0/z, [%[b]]\n\t"
            "st1w z0.s, p0, [%[z0o]]\n\t"            /* 调试: 存 z0 确认 ld1w 生效 */
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s\n\t"
            "mov w12, #0\n\t"
            "mova z2.s, p0/m, za0h.s[w12, 0]\n\t"
            "st1w z2.s, p0, [%[zao]]\n\t"
            "smstop za\n\t"
            "smstop sm\n\t"
            :
            : [a] "r" (a), [b] "r" (b), [z0o] "r" (z0_out), [zao] "r" (za_out)
            : "memory", "w12"
        );
        ok = 1;
    }
    restore_handler(&old);
    return ok;
}

static void print_arr(const char *name, const float *a, const float *expected) {
    int correct = 1;
    printf("%s:\n", name);
    for (int i = 0; i < 16; i++) {
        int exp = (int)(expected[i] + 0.5f);
        int got = (int)(a[i] + 0.5f);
        printf("  [%2d] = %8.3f  (expected %2d)  %s\n",
               i, (double)a[i], exp, got == exp ? "ok" : "MISMATCH");
        if (got != exp) correct = 0;
    }
    printf("→ %s\n\n", correct ? "CORRECT" : "WRONG");
}

int main(void) {
    float ones[16]      = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    float seq_1_16[16]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    float min_out[16] = {0};
    float idx_out[16] = {0};
    float ld_out[16]  = {0};
    float z0_out[16]  = {0};
    float za_out[16]  = {0};

    printf("=== fmopa sanity v4: 4-layer diagnosis ===\n\n");

    int m_ok = test_minimal(min_out);
    printf("[test1] minimal (fmov z0=#1.0): %s\n", m_ok ? "executed" : "SIGILL");
    if (m_ok) print_arr("  z0 (expect all 1)", min_out, ones);

    int i_ok = test_index(idx_out);
    printf("[test2] index (index z0, #1, #1): %s\n", i_ok ? "executed" : "SIGILL");
    if (i_ok) print_arr("  z0 (expect 1..16)", idx_out, seq_1_16);

    int l_ok = test_ld1w(seq_1_16, ld_out);
    printf("[test3] ld1w (load from memory): %s\n", l_ok ? "executed" : "SIGILL");
    if (l_ok) print_arr("  z0 (expect 1..16)", ld_out, seq_1_16);

    int f_ok = test_fmopa_ld1w(seq_1_16, seq_1_16, z0_out, za_out);
    printf("[test4] fmopa (ld1w + fmopa + mova): %s\n", f_ok ? "executed" : "SIGILL");
    if (f_ok) {
        print_arr("  z0 (expect 1..16)", z0_out, seq_1_16);
        print_arr("  za0 slice 0 (expect 1..16)", za_out, seq_1_16);
    }

    printf("--- 诊断 ---\n");
    if (m_ok && l_ok && f_ok) {
        int mc=1, lc=1, fc=1, zc=1;
        for (int i = 0; i < 16; i++) {
            if ((int)(min_out[i]+0.5f) != 1) mc=0;
            if ((int)(ld_out[i]+0.5f) != i+1) lc=0;
            if ((int)(z0_out[i]+0.5f) != i+1) zc=0;
            if ((int)(za_out[i]+0.5f) != i+1) fc=0;
        }
        printf("test1 fmov: %s\n", mc ? "ok" : "WRONG");
        printf("test2 index: %s (index 指令问题, 但 microkernel 用 ld1w 不影响)\n",
               i_ok ? "executed but WRONG" : "SIGILL");
        printf("test3 ld1w: %s\n", lc ? "ok" : "WRONG");
        printf("test4 fmopa z0: %s\n", zc ? "ok" : "WRONG");
        printf("test4 fmopa za: %s\n", fc ? "ok" : "WRONG");
        if (lc && fc) {
            printf("\n✓ ld1w + fmopa + mova 全对 → 可推进 micro/primary.c (用 ld1w 加载, 跳过 index)\n");
        }
    }

    return 0;
}
