/*
 * fmopa_sanity.c — fmopa 计算正确性 + ZA slice read 语法验证 (v2: 带 z0 调试)
 *
 * 测试逻辑:
 *   z0 = [1, 2, ..., 16]   (index z0.s, #1, #1)
 *   z1 = [1, 2, ..., 16]   (index z1.s, #1, #1)
 *   fmopa za0 → za0[i][j] = (i+1)*(j+1)
 *   mova za0h slice 0 → 期望 [1, 2, ..., 16]
 *
 * v2 加调试: fmopa 前先存 z0 出来, 确认 index 在 streaming 模式内生效
 *   - 若 z0 = [1..16] 但 za 全 0 → 问题在 fmopa 或 mova
 *   - 若 z0 也全 0 → 问题在 index 或 st1w (streaming 内 SVE 指令链路)
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

__attribute__((target("sve,sme")))
static int fmopa_sanity_test(float *z0_out, float *za_out) {
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
            "st1w z0.s, p0, [%[z0o]]\n\t"            /* 调试: 存 z0 确认 index 生效 */
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s\n\t"
            "mov w12, #0\n\t"
            "mova z2.s, p0/m, za0h.s[w12, 0]\n\t"     /* 读 za0 水平 slice 0 */
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

int main(void) {
    float z0_out[16] = {0};
    float za_out[16] = {0};
    printf("=== fmopa sanity test (v2: with z0 debug) ===\n\n");

    int ok = fmopa_sanity_test(z0_out, za_out);
    printf("executed: %s\n\n", ok ? "yes" : "SIGILL");

    if (!ok) {
        printf("SIGILL — ZA slice 语法或指令顺序问题\n");
        return 1;
    }

    /* 检查 z0 */
    int z0_correct = 1;
    printf("z0 (expected 1..16, 验证 index 在 streaming 内生效):\n");
    for (int i = 0; i < 16; i++) {
        int expected = i + 1;
        int got = (int)(z0_out[i] + 0.5f);
        printf("  [%2d] = %8.3f  (expected %2d)  %s\n",
               i, (double)z0_out[i], expected, got == expected ? "ok" : "MISMATCH");
        if (got != expected) z0_correct = 0;
    }
    printf("z0: %s\n\n", z0_correct ? "CORRECT" : "WRONG");

    /* 检查 za slice */
    int za_correct = 1;
    printf("za0 horizontal slice 0 (expected 1..16):\n");
    for (int i = 0; i < 16; i++) {
        int expected = i + 1;
        int got = (int)(za_out[i] + 0.5f);
        printf("  [%2d] = %8.3f  (expected %2d)  %s\n",
               i, (double)za_out[i], expected, got == expected ? "ok" : "MISMATCH");
        if (got != expected) za_correct = 0;
    }
    printf("za: %s\n\n", za_correct ? "CORRECT" : "WRONG");

    /* 诊断结论 */
    printf("--- 诊断 ---\n");
    if (z0_correct && za_correct) {
        printf("ALL CORRECT → fmopa 语义 + ZA slice read 语法全对, 可推进 micro/primary.c\n");
    } else if (!z0_correct) {
        printf("z0 WRONG → index 或 st1w 在 streaming 模式内未生效\n");
        printf("           可能 ptrue/index/st1w 链路有问题\n");
    } else if (z0_correct && !za_correct) {
        printf("z0 ok 但 za WRONG → fmopa 没累加到 za0, 或 mova 读错 tile/slice\n");
        printf("           可能: fmopa 谓词/操作数顺序, 或 mova 的 w12/slice 索引语义\n");
    }

    return (z0_correct && za_correct) ? 0 : 1;
}
