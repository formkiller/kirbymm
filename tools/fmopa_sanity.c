/*
 * fmopa_sanity.c — fmopa 计算正确性 + ZA slice read 语法验证
 *
 * 目的: 在写完整 microkernel 前, 验证 fmopa 累加语义 + ZA slice read 汇编语法正确
 *
 * 测试逻辑:
 *   z0 = [1, 2, 3, ..., 16]   (index z0.s, #1, #1)
 *   z1 = [1, 2, 3, ..., 16]   (index z1.s, #1, #1)
 *   fmopa za0.s, p0/m, p0/m, z0.s, z1.s
 *     → za0[i][j] = z0[i] * z1[j] = (i+1) * (j+1)
 *   mova z2.s, p0/m, za0h.s[0]   (读 za0 水平 slice 0 = 第 0 行)
 *     → z2 = [za0[0][0], za0[0][1], ..., za0[0][15]]
 *            = [1*1, 1*2, ..., 1*16] = [1, 2, ..., 16]
 *   st1w z2 → out[0..15]
 *   期望 out = [1, 2, ..., 16]
 *
 * 若 out 正确: fmopa 语义对, ZA slice read 语法对 → 可推进 micro/primary.c
 * 若 SIGILL: ZA slice 语法可能错, 需调整 mova 语法
 *
 * 编译: clang -std=c11 -O2 -arch arm64 fmopa_sanity.c -o fmopa_sanity
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
static int fmopa_sanity_test(float *out) {
    struct sigaction old = install_handler();
    g_sigill = 0;
    int ok = 0;
    if (sigsetjmp(g_jmp, 1) == 0) {
        __asm__ volatile(
            "smstart sm\n\t"                          /* 进入 streaming SVE 模式 */
            "smstart za\n\t"                          /* 启用 ZA 阵列 */
            "zero za\n\t"                             /* 清零整个 ZA */
            "ptrue p0.s, all\n\t"                     /* 谓词 p0 全真 */
            "index z0.s, #1, #1\n\t"                  /* z0 = [1,2,...,16] */
            "index z1.s, #1, #1\n\t"                  /* z1 = [1,2,...,16] */
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s\n\t" /* za0[i][j] += (i+1)*(j+1) */
            "mova z2.s, p0/m, za0h.s[0]\n\t"          /* 读 za0 水平 slice 0 (第0行) */
            "st1w z2.s, p0, [%[out]]\n\t"             /* 存 16 个 FP32 到 out */
            "smstop za\n\t"
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

int main(void) {
    float out[16] = {0};
    printf("=== fmopa sanity test ===\n\n");

    int ok = fmopa_sanity_test(out);
    printf("fmopa + ZA slice read: %s\n\n", ok ? "executed" : "SIGILL");

    int correct = 0;
    if (ok) {
        printf("za0 horizontal slice 0 (expected 1..16):\n");
        correct = 1;
        for (int i = 0; i < 16; i++) {
            int expected = i + 1;
            int got = (int)(out[i] + 0.5f);
            printf("  [%2d] = %6.2f  (expected %2d)  %s\n",
                   i, (double)out[i], expected, got == expected ? "ok" : "MISMATCH");
            if (got != expected) correct = 0;
        }
        printf("\nresult: %s\n", correct ? "CORRECT" : "WRONG");
        if (correct) {
            printf("\nif CORRECT: fmopa 计算语义正确, ZA slice read 语法正确\n");
            printf("            → 可推进 micro/primary.c 完整 microkernel\n");
        }
    } else {
        printf("SIGILL — 可能 ZA slice 语法不对, 或指令顺序问题\n");
        printf("需调整 mova 语法 (如 za0h.s[w,0] 或 za0h.s[0,0])\n");
    }

    return (ok && correct) ? 0 : 1;
}
