# KirbyMM 复现项目骨架设计

> 状态: 设计层 (未写实现代码)
> 依据: M4 探测结论 (SVL=64B, vl=16, fmopa 可用, 与论文一致)

## 1. 探测结论 (已完成, 作为参数表依据)

| 参数 | 值 | 来源 |
|---|---|---|
| SVL | 64 bytes = 512-bit | cntb 实测 |
| vl (FP32) | 16 | cntw 实测 |
| fmopa 可用 | yes | v3 sanity check |
| BiReg-CMR 解 | mr1=32, nr=32, mr2=0 | Eq.1 求解, CMR=512 |
| ZA 瓦片 | 4 个 16x16 | 论文 §II-A |
| 主例程 C 分块 | 32x32 (2x2 瓦片块) | Fig.7 |

关键约束: M4 的 SVE/SME 指令只在 streaming 模式内可用, 所有 MicroKernel 代码必须包在 `smstart sm` ... `smstop sm` 之间.

## 2. 目录结构

```
kirbymm/
├── CMakeLists.txt          # 顶层构建
├── DESIGN.md               # 本文件
├── include/                # 公共头
│   ├── kirby.h             # 对外入口接口
│   ├── params.h            # 参数表 (vl/mr1/nr/mr2/blkm/...)
│   └── types.h             # microkernel 函数指针类型
├── platform/               # 平台抽象层
│   ├── sme_wrap.h          # smstart/smstop 宏 + target attribute 封装
│   ├── detect.h            # SVL/SME 探测接口
│   └── detect.c            # 复用 svl_probe 逻辑
├── micro/                  # MicroKernel (L4-L5)
│   ├── primary.c           # 主例程: 纯 fmopa, 32x32 块
│   └── dedicated.c         # 专用例程: fmopa + fmla 交错 (边界情况)
├── macro/                  # MacroKernel (L1-L3)
│   ├── partition.c         # 分块参数求解 (Eq.5/6/7)
│   └── loop.c              # 五层循环主体
├── packing/                # 数据打包
│   └── pack.c              # ZA 瓦片打包 (隐式转置)
├── test/                   # 测试与验证
│   ├── naive.c             # naive GEMM (正确性基准)
│   ├── verify.c           # 逐元素比对
│   └── bench.c             # 性能测量 (对照 Roofline)
└── tools/
    └── svl_probe.c         # 已完成: SVL/SME 探测
```

## 3. 核心接口签名

```c
/* include/kirby.h */
void kirby_sgemm_fp32(int M, int N, int K,
                      const float *A, const float *B, float *C);

/* include/types.h */
typedef void (*microkernel_fp32_t)(
    float       *C,    // [blkm x blkn] 输出, 已分配
    const float *A,    // [blkm x blkk] 打包后
    const float *B,    // [blkk x blkn] 打包后
    int blkm, int blkn, int blkk);

/* include/params.h — 平台参数表 */
typedef struct {
    int vl;        // FP32 元素/向量, M4=16
    int mr1, mr2, nr;  // BiReg-CMR 解
    int blkm, blkn, blkk;  // MacroKernel 分块 (待 Eq.6/7 求解)
    int si, sj;    // MicroKernel 内步长
    int za_tiles;  // ZA 瓦片数, FP32=4
} kirby_params_t;

extern const kirby_params_t KIRBY_PARAMS_M4;
extern const kirby_params_t KIRBY_PARAMS_LX2;  // 阶段二填
```

## 4. 平台抽象层

```c
/* platform/sme_wrap.h */
#define KIRBY_SME_BEGIN()  __asm__ volatile("smstart sm\n\t" ::: "memory")
#define KIRBY_SME_END()    __asm__ volatile("smstop sm\n\t" ::: "memory")
#define KIRBY_ZA_BEGIN()   __asm__ volatile("smstart za\n\t" ::: "memory")
#define KIRBY_ZA_END()     __asm__ volatile("smstop za\n\t" ::: "memory")

/* 所有 MicroKernel 函数标注 */
#define KIRBY_MICRO_FN     __attribute__((target("sve,sme")))
```

MicroKernel 函数模板:
```c
KIRBY_MICRO_FN
void primary_routine_fp32(float *C, const float *A, const float *B,
                          int blkm, int blkn, int blkk) {
    KIRBY_ZA_BEGIN();      // 启用 ZA
    KIRBY_SME_BEGIN();     // 进入 streaming 模式
    // ... 4 条 fmopa 循环 ...
    KIRBY_SME_END();
    KIRBY_ZA_END();
}
```

## 5. 参数表 (M4, vl=16)

```c
const kirby_params_t KIRBY_PARAMS_M4 = {
    .vl = 16,
    .mr1 = 32, .mr2 = 0, .nr = 32,   // 主例程 BiReg-CMR 解
    .blkm = 0, .blkn = 0, .blkk = 0, // 待 Eq.6/7 数值求解 (需 M4 L1/L2 缓存参数)
    .si = 32, .sj = 32,              // 主例程简化: si=mr1, sj=nr
    .za_tiles = 4,
};
```

blkm/blkn/blkk 待办: 实测 M4 的 L1/L2 缓存几何 (组数/相联度/行大小), 代入 Eq.6/7 联立求解.

## 6. 验证策略 (三层)

| 层 | 方法 | 工具 |
|---|---|---|
| 数值正确性 | naive GEMM vs KirbyMM, 逐元素 abs err < 1e-4 | test/verify.c |
| 性能 | 对照 M4 Roofline (P_peak=0.512 TFLOPS), 报告占比 | test/bench.c, mach_absolute_time |
| 缓存 | Instruments 测 L1 miss (M4 上 load 命中率 N/A, 仅 miss 计数) | Xcode Instruments |

## 7. CMake 结构 (顶层)

```cmake
cmake_minimum_required(VERSION 3.20)
project(kirbymm C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -arch arm64")
    # M4: 不需 -march, 函数级 target attribute 放行 SME/SVE
elseif(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=aarch64-linux-gnu -march=armv9-a+sme")
endif()

add_library(kirbymm
    platform/detect.c
    macro/partition.c macro/loop.c
    packing/pack.c
    micro/primary.c micro/dedicated.c
)
target_include_directories(kirbymm PUBLIC include)

add_executable(svl_probe tools/svl_probe.c)
add_executable(naive_gemm test/naive.c)
add_executable(verify test/verify.c)
add_executable(bench test/bench.c)
target_link_libraries(verify kirbymm)
target_link_libraries(bench kirbymm)
```

## 8. 开发顺序 (建议)

1. naive.c (纯 C 三重循环) — 正确性基准, M4 立即可跑
2. platform/detect.c — 整合 svl_probe, 运行时填充 params
3. micro/primary.c — 主例程 4 条 fmopa, 对照 naive 验证
4. packing/pack.c — ZA 瓦片打包
5. macro/loop.c + partition.c — 五层循环 + 分块参数求解
6. micro/dedicated.c — 专用例程 (fmopa + fmla 交错)
7. bench.c — 性能测量

阶段二 (LX2) 接入:
- 重新跑 detect.c 填 LX2 参数表
- 启用 dedicated.c 的 overlap 段 (条件编译)
- 对照 KBLAS 性能
