/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <sstream>
#include <gtest/gtest.h>
#include <chrono>

#include "../../../include/utils/utility.h"
using namespace std;

class CompilePerfTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "CompilePerfTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "CompilePerfTest TearDown" << endl;
    }

    static std::string GetIncludePath()
    {
        static const string ascendIncludePathPrefix = string("-I") + ASCEND_DIR + "/";
        static const std::vector<string> ascendIncludeSubDirs = {
            "include",
            "compiler/tikcpp/include",
            "compiler/ascendc/include/basic_api/impl",
            "compiler/ascendc/include/basic_api/interface",
            "compiler/ascendc/include/highlevel_api/impl",
            "compiler/ascendc/include/highlevel_api/tiling",
            "compiler/ascendc/impl/aicore/basic_api",
        };
        static const string projectPath = PROJECT_DIR;
        stringstream ss;
        for (const auto& includePath : ascendIncludeSubDirs) {
            ss << ascendIncludePathPrefix << includePath << " ";
        }
        ss << "-I" << projectPath << "/include" << " -I" << projectPath << "/examples/common";
        return ss.str();
    }

    static std::string GetCompileCmd(const std::string& fileName)
    {
        string srcFilePath = string(CASES_DIR) + "/" + fileName;
        static const string includePath = GetIncludePath();
        static const string compileOptions =
            // "-ftime-trace -ftime-trace-granularity=0 "   //
            // 放开后在程序执行目录有性能统计的json，可在edge://tracing/中分析
            "-g -fPIE -fdiagnostics-color=always -O3 -w -std=gnu++17 "
            "--npu-arch=dav-2201 "
            "-xasc";
        stringstream ss;
        ss << "bisheng " << includePath << " ";
        ss << compileOptions << " ";
        ss << "-MD -MT " << BINARY_DIR << "/" << fileName << ".o ";
        ss << "-MF " << BINARY_DIR << "/" << fileName << ".o.d ";
        ss << "-o " << BINARY_DIR << "/" << fileName << ".o.o ";
        ss << "-c " << srcFilePath;
        return ss.str();
    }

    static inline bool ExecCompileCmd(const std::string& cmd)
    {
        auto result = system(cmd.c_str());
        if (result != 0) {
            std::cerr << "compile failed: " << cmd << std::endl;
            return false;
        }
        return true;
    }

    static uint64_t GetCompileTime(const std::string& fileName)
    {
        const auto cmd = GetCompileCmd(fileName);
        std::cout << cmd << std::endl;
        auto start = std::chrono::steady_clock::now();
        auto result = ExecCompileCmd(cmd);
        auto end = std::chrono::steady_clock::now();
        if (!result) {
            std::cerr << "compile failed: " << fileName << std::endl;
            return -1;
        }
        auto usedTime = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        std::cout << "compile time:" << usedTime << " s" << std::endl;
        return usedTime;
    }
};

TEST_F(CompilePerfTest, long_expr_op)
{
    auto time = GetCompileTime("120_expr_op.cpp"); // 120个expression的操作测试
    EXPECT_LE(time, 90);                           // 120个节点 wsl 72-81s，linux workspace 77-81s, 偶尔能上到114s
}

TEST_F(CompilePerfTest, long_expr_op_with_bind_buff)
{
    auto time = GetCompileTime("120_expr_op_with_bind_buff.cpp"); // 120个expression的操作测试
    EXPECT_LE(time, 80);                                          // 60个节点 wsl 68s
}

TEST_F(CompilePerfTest, long_expr_op_with_auto)
{
    auto time = GetCompileTime("120_expr_op_with_auto.cpp"); // 120个expression的操作测试
    EXPECT_LE(time, 50);                                     // 60个节点 wsl 48s
}

TEST_F(CompilePerfTest, expr_linearizer_perf)
{
    auto time = GetCompileTime("expr_linearizer_perf.cpp"); // 120个expression的操作测试
    EXPECT_LE(time, 50);                                    // wsl 35s
}
