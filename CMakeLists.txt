# CMakeLists.txt — KirbyMM 顶层构建
# 阶段一最小集: naive GEMM + 正确性验证 + SVL 探测工具
# 后续 (待 kirbymm 库实现) 见文件末注释块

cmake_minimum_required(VERSION 3.20)
project(kirbymm C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 优化级别
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# 平台适配
#   macOS (Apple M4): 用 -arch arm64, 不需全局 -march
#     SME/SVE 指令靠函数级 __attribute__((target("sve,sme"))) 放行
#   Linux (ARMv9/LX2): 用 -march=armv9-a+sme
if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -arch arm64")
elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv9-a+sme")
endif()

# 公共头路径
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---- naive GEMM (正确性基准, 纯 C) ----
add_library(naive STATIC test/naive.c)

# ---- SVL/SME 探测工具 (已完成) ----
add_executable(svl_probe tools/svl_probe.c)

# ---- fmopa 计算正确性 + ZA slice read 语法验证 ----
add_executable(fmopa_sanity tools/fmopa_sanity.c)

# ---- M4 缓存几何探测 (§V-A 内存敏感分块前置) ----
add_executable(cache_probe tools/cache_probe.c)

# ---- 正确性验证 (ref=naive, test=kirby) ----
add_executable(verify test/verify.c)
target_link_libraries(verify PRIVATE kirbymm m)

# ---- 性能基准 (naive vs kirby GFLOPS 对照) ----
add_executable(bench test/bench.c)
target_link_libraries(bench PRIVATE kirbymm naive)

# ---- microkernel 裸性能定标 (隔离调度层, 修正 peak 口径) ----
add_executable(kernel_bench tools/kernel_bench.c)
target_link_libraries(kernel_bench PRIVATE kirbymm)

# ---- KirbyMM 库 (主例程: 16x16 单瓦片 + 32x32 4-tile + MacroKernel 块遍历) ----
add_library(kirbymm micro/primary.c)
target_link_libraries(kirbymm PUBLIC naive)
