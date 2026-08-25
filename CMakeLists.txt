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

# ---- 正确性验证工具 ----
add_executable(verify test/verify.c)
target_link_libraries(verify PRIVATE naive m)

# ---- 后续: KirbyMM 库 (待实现) ----
# 取消注释并填充源文件即可接入:
#
# add_library(kirbymm
#     platform/detect.c
#     macro/partition.c
#     macro/loop.c
#     packing/pack.c
#     micro/primary.c
#     micro/dedicated.c
# )
# target_include_directories(kirbymm PUBLIC include)
#
# # 用 KirbyMM 替换 naive 作为被测实现
# # verify.c 里把 test = naive_sgemm_fp32 改成 test = kirby_sgemm_fp32
# target_link_libraries(verify PRIVATE kirbymm m)
#
# # 性能基准
# add_executable(bench test/bench.c)
# target_link_libraries(bench PRIVATE kirbymm naive m)
