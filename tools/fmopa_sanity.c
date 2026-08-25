/*
 * fmopa_sanity.c — 三层诊断 streaming SVE/SME 链路 (v3)
 *
 * 诊断分层:
 *   1. minimal: smstart sm + ptrue + fmov z0=#1.0 + st1w
 *      期望 out=[1,1,...,1]  → 验证 ptrue/fmov/st1w 基础链路
 *   2. index:   smstart sm + ptrue + index z0 + st1w
 *      期望 out=[1,2,...,16] → 验证 index 指令
 *   3. fmopa:   smstart sm+za + ptrue + index + fmopa + mova + st1w
 *      期望 za=[1,2,...,16]  → 验证 fmopa + ZA slice read
 *
 * 定位逻辑:
 *   - test1 错 → ptrue 或 st1w 在 streaming 内根本不工作
 *   - test1 对 test2 错 → index 指令问题
 *   - test1+2 对 test3 错 → fmopa 或 mova 问题
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

/* test 1: minimal — ptrue + fmov z0=#1.0 + st1w */
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

/* test 2: index — ptrue + index z0 + st1w */
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

/* test 3: fmopa — ptrue + index + fmopa + mova + st1w */
__attribute__((target("sve,sme")))
static int test_fmopa(float *z0_out, float *za_out) {
    struct sigaction old = install_handler();
    g_sigill = 0;
    int ok = 0;
    if (sigsetjmp(g_jmp, 1) == 0) {
        __asm__ volatile(
            "smstart sm\n\t"
            "smstart za\n\t"
            "zero {za}\n\t"
            "ptrue p0.s, all\n\t"
            "index z0.s, #1, #1\n\t"
            "index z1.s, #1, #1\n\t"
            "st1w z0.s, p0, [%[z0o]]\n\t"
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s\n\t"
            "mov w12, #0\n\t"
            "mova z2.s, p0/m, za0h.s[w12, 0]\n\t"
            "st1w z2.s, p0, [%[zao]]\n\t"
            "smstop za\n\t"
            "smstop sm\n\t"
            :
            : [z0o] "r" (z0_out), [zao] "r" (za_out)
            : "memory", "w12"
        );
        ok = 1;
    }
    restore_handler(&old);
    return ok;
}

static void print_result(const char *name, const float *a, int expect_start) {
    int correct = 1;
    int expect_end = expect_start + 15;
    printf("%s (expected %d..%d):\n", name, expect_start, expect_end);
    for (int i = 0; i < 16; i++) {
        int expected = expect_start + i;
        int got = (int)(a[i] + 0.5f);
        printf("  [%2d] = %8.3f  (expected %2d)  %s\n",
               i, (double)a[i], expected, got == expected ? "ok" : "MISMATCH");
        if (got != expected) correct = 0;
    }
    printf("→ %s: %s\n\n", name, correct ? "CORRECT" : "WRONG");
}

int main(void) {
    float min_out[16] = {0};
    float idx_out[16] = {0};
    float z0_out[16] = {0};
    float za_out[16] = {0};

    printf("=== fmopa sanity v3: 3-layer diagnosis ===\n\n");

    /* test 1: minimal */
    int m_ok = test_minimal(min_out);
    printf("[test1] minimal (ptrue+fmov+st1w): %s\n", m_ok ? "executed" : "SIGILL");
    if (m_ok) print_result("  z0 (expect 1,1,...,1)", min_out, 1);

    /* test 2: index */
    int i_ok = test_index(idx_out);
    printf("[test2] index (ptrue+index+st1w): %s\n", i_ok ? "executed" : "SIGILL");
    if (i_ok) print_result("  z0 (expect 1..16)", idx_out, 1);

    /* test 3: fmopa */
    int f_ok = test_fmopa(z0_out, za_out);
    printf("[test3] fmopa (full SME): %s\n", f_ok ? "executed" : "SIGILL");
    if (f_ok) {
        print_result("  z0 (expect 1..16)", z0_out, 1);
        print_result("  za0 slice 0 (expect 1..16)", za_out, 1);
    }

    printf("--- 诊断 ---\n");
    if (!m_ok) {
        printf("test1 SIGILL: smstart/ptrue/fmov/st1w 基础链路不通\n");
    } else {
        int min_correct = 1;
        for (int i = 0; i < 16; i++)
            if ((int)(min_out[i] + 0.5f) != 1) min_correct = 0;
        if (!min_correct) {
            printf("test1 WRONG: ptrue 没让 p0 全真, st1w 被空谓词屏蔽\n");
            printf("          → 检查 ptrue 语法或谓词寄存器初始化\n");
        } else {
            printf("test1 ok: ptrue+fmov+st1w 链路正常\n");
            if (!i_ok) {
                printf("test2 SIGILL: index 指令问题\n");
            } else {
                int idx_correct = 1;
                for (int i = 0; i < 16; i++)
                    if ((int)(idx_out[i] + 0.5f) != i + 1) idx_correct = 0;
                if (!idx_correct) {
                    printf("test2 WRONG: index 生成的值不对\n");
                } else {
                    printf("test2 ok: index 正常\n");
                    if (!f_ok) {
                        printf("test3 SIGILL: fmopa/mova/za 链路问题\n");
                    } else {
                        int z0c = 1, zac = 1;
                        for (int i = 0; i < 16; i++) {
                            if ((int)(z0_out[i] + 0.5f) != i + 1) z0c = 0;
                            if ((int)(za_out[i] + 0.5f) != i + 1) zac = 0;
                        }
                        if (!z0c) printf("test3 z0 WRONG: streaming 内 index 在加 za 后异常\n");
                        else if (!zac) printf("test3 za WRONG: fmopa 没累加或 mova 读错\n");
                        else printf("test3 ok: fmopa + ZA slice read 全对, 可推进 micro/primary.c\n");
                    }
                }
            }
        }
    }

    return 0;
}
