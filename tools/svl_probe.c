/*
 * svl_probe.c — KirbyMM SVL 读取 (v4)
 *
 * v3 结论: fmopa 可用, ptrue 可用, 但 rdvl SIGILL (Apple M4 SVE 子集缺 rdvl)
 * v4 策略: 换用 SVE count 指令族读 SVL
 *   cntb Xd  → VL/8 字节 (streaming 内 = SVL 字节数, 最直接)
 *   cntw Xd  → VL/32 (streaming 内 = SVL/32 = FP32 元素数 = vl)
 *   cnth Xd  → VL/16
 *   cntd Xd  → VL/64
 * 语法: cnt{b,h,w,d} Xd, pattern, modifier (pattern=ALL, modifier=MUL 可省略)
 *
 * 编译:
 *   macOS: clang -std=c11 -O2 -arch arm64 svl_probe.c -o svl_probe
 *   fallback: clang -std=c11 -O2 -arch arm64 -mattr=+sve,+sme,+sme2 svl_probe.c -o svl_probe
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

#define PROBE_VAL(asm_block)                                   \
    struct sigaction old = install_handler();                  \
    g_sigill = 0;                                              \
    long result = -1;                                          \
    if (sigsetjmp(g_jmp, 1) == 0) {                            \
        uint64_t v = 0;                                        \
        __asm__ volatile(asm_block : "=r"(v) : : "memory");    \
        result = (long)v;                                      \
    }                                                          \
    restore_handler(&old);                                     \
    return result;

#define PROBE_INT(asm_block)                                   \
    struct sigaction old = install_handler();                  \
    g_sigill = 0;                                              \
    int ok = 0;                                                \
    if (sigsetjmp(g_jmp, 1) == 0) {                            \
        __asm__ volatile(asm_block : : : "memory");           \
        ok = 1;                                                \
    }                                                          \
    restore_handler(&old);                                     \
    return ok;

/* 确认 fmopa 可用 (v3 已测, 保留作 sanity check) */
__attribute__((target("sme")))
static int probe_fmopa(void) {
    PROBE_INT(
        "smstart sm\n\t"
        "smstart za\n\t"
        "fmopa za0.s, p0/m, p0/m, z0.s, z0.s\n\t"
        "smstop za\n\t"
        "smstop sm\n\t"
    );
}

/* streaming 内 cntb → SVL 字节数 (主探测) */
__attribute__((target("sve,sme")))
static long probe_cntb_streaming(void) {
    PROBE_VAL(
        "smstart sm\n\t"
        "cntb %0\n\t"
        "smstop sm\n\t"
    );
}

/* streaming 内 cntw → SVL/32 = FP32 元素数 = vl */
__attribute__((target("sve,sme")))
static long probe_cntw_streaming(void) {
    PROBE_VAL(
        "smstart sm\n\t"
        "cntw %0\n\t"
        "smstop sm\n\t"
    );
}

/* streaming 内 cnth → SVL/16 */
__attribute__((target("sve,sme")))
static long probe_cnth_streaming(void) {
    PROBE_VAL(
        "smstart sm\n\t"
        "cnth %0\n\t"
        "smstop sm\n\t"
    );
}

/* streaming 内 cntd → SVL/64 */
__attribute__((target("sve,sme")))
static long probe_cntd_streaming(void) {
    PROBE_VAL(
        "smstart sm\n\t"
        "cntd %0\n\t"
        "smstop sm\n\t"
    );
}

static const char *tf(long v) { return v ? "yes" : "no(SIGILL)"; }

int main(void) {
    printf("=== KirbyMM SVL probe (v4: count instrs) ===\n\n");

    int fmopa_ok = probe_fmopa();
    printf("[fmopa]            : %s  (sanity check)\n", tf(fmopa_ok));
    if (!fmopa_ok) {
        printf("SME compute not available, abort.\n");
        return 1;
    }

    long cntb = probe_cntb_streaming();
    long cntw = probe_cntw_streaming();
    long cnth = probe_cnth_streaming();
    long cntd = probe_cntd_streaming();

    printf("[cntb] streaming   : %s\n", tf(cntb > 0));
    if (cntb > 0) printf("      SVL = %ld bytes = %ld bits\n", cntb, cntb * 8);
    printf("[cntw] streaming   : %s\n", tf(cntw > 0));
    if (cntw > 0) printf("      vl (FP32 elems) = %ld\n", cntw);
    printf("[cnth] streaming   : %s\n", tf(cnth > 0));
    if (cnth > 0) printf("      SVL/16 = %ld\n", cnth);
    printf("[cntd] streaming   : %s\n", tf(cntd > 0));
    if (cntd > 0) printf("      SVL/64 = %ld\n", cntd);

    /* 推算 SVL (用任一可用的 count) */
    long svl_bytes = -1;
    if (cntb > 0) svl_bytes = cntb;
    else if (cntw > 0) svl_bytes = cntw * 4;
    else if (cnth > 0) svl_bytes = cnth * 2;
    else if (cntd > 0) svl_bytes = cntd * 8;

    printf("\n--- SVL determination ---\n");
    if (svl_bytes <= 0) {
        printf("all count instrs SIGILL — cannot read SVL\n");
        printf("but fmopa works, SME is available; SVL must be obtained another way\n");
        return 1;
    }

    long vl_fp32 = svl_bytes / 4;
    printf("SVL = %ld bytes = %ld bits\n", svl_bytes, svl_bytes * 8);
    printf("vl (FP32 elems/vector) = %ld\n", vl_fp32);

    printf("\n--- compare with paper ---\n");
    printf("paper: vl=16, SVL=64 bytes=512-bit, ZA=4 x 16x16 tiles\n");
    if (svl_bytes == 64) {
        printf("[MATCH] machine SVL = 64 bytes, matches paper\n");
        printf("        use paper BiReg-CMR solution: mr1=32, nr=32, mr2=0 (CMR=512)\n");
        printf("        ZA tiles: 4 x 16x16, 2x2 block covers C 32x32\n");
    } else {
        printf("[MISMATCH] machine SVL = %ld, paper = 64\n", svl_bytes);
        printf("           re-solve Eq.1 with vl=%ld:\n", vl_fp32);
        printf("             a=mr1/vl, b=nr/vl, a*b<=4, maximize 2*vl^2*a*b/(a+b)\n");
    }

    printf("\n--- reproduction readiness ---\n");
    printf("fmopa: %s, SVL read: %s\n",
           fmopa_ok ? "available" : "N/A",
           svl_bytes > 0 ? "success" : "failed");
    if (fmopa_ok && svl_bytes > 0) {
        printf("verdict: READY to proceed with KirbyMM reproduction\n");
    } else if (fmopa_ok) {
        printf("verdict: SME works but SVL unknown, proceed with caution (assume paper default 64B?)\n");
    }

    printf("\n=== probe done ===\n");
    return 0;
}
