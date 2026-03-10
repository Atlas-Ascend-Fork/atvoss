/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file log.h
 * \brief
 */
#ifndef ATVOSS_LOG_H_
#define ATVOSS_LOG_H_

#define DO_JOIN_SYMBOL(symbol1, symbol2) symbol1##symbol2
#define JOIN_SYMBOL(symbol1, symbol2) DO_JOIN_SYMBOL(symbol1, symbol2)

#ifdef __COUNTER__
#define UNIQUE_ID __COUNTER__
#else
#define UNIQUE_ID __LINE__
#endif

#define UNIQUE_NAME(prefix) JOIN_SYMBOL(prefix, UNIQUE_ID)

#ifdef __CCE_KT_TEST__
#define RUN_LOG_BASE(...)     \
    if (GetBlockIdx() == 0) { \
        printf(__VA_ARGS__);  \
    }
#else
#define RUN_LOG_BASE(...)    \
    do {                     \
        printf(__VA_ARGS__); \
    } while (0)
#endif

#include <cxxabi.h>
#include <securec.h>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

template <typename T>
std::string GetTypeName()
{
    std::string result;
    const char* name = typeid(T).name();
    int status = -1;
    std::unique_ptr<char, void (*)(void*)> demangleName{
        abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};

    if (status == 0 && demangleName) {
        result = demangleName.get();
    } else {
        result = "[mangle]" + std::string(name);
    }
    return result;
}

class TemplatePrettyFormater {
private:
    struct FormatConfig {
        int indentSize = 4;
        int maxLineLength = 120;
        int maxParamsPerLine = 3;
        int maxTemplateDepthPerLine = 2;
    };

    static std::string CreateIndent(int level, const FormatConfig& config)
    {
        return std::string(level * config.indentSize, ' ');
    }

    static std::vector<std::string> SplitTemplateParams(const std::string& paramsStr)
    {
        std::vector<std::string> params;
        std::string current;
        int angleDepth = 0;
        for (auto c : paramsStr) {
            if (c == '<') {
                angleDepth++;
                current += c;
            } else if (c == '>') {
                angleDepth--;
                current += c;
            } else if (c == ',' && angleDepth == 0) {
                params.push_back(TrimString(current));
                current.clear();
            } else {
                current += c;
            }
        }

        if (!current.empty()) {
            params.push_back(TrimString(current));
        }

        return params;
    }

    static std::string TrimString(const std::string& str)
    {
        size_t start = str.find_first_not_of(" \t\r\n");
        size_t end = str.find_last_not_of(" \t\r\n");

        if (start == std::string::npos) {
            return "";
        }

        return str.substr(start, end - start + 1);
    }

    static bool IsComplexTemplate(const std::string& param)
    {
        int maxDepth = GetTemplateDepth(param);
        return maxDepth > 1 || param.find('<') != param.find_last_of('<');
    }

    static int GetTemplateDepth(const std::string& str)
    {
        int depth = 0;
        int maxDepth = 0;
        for (auto c : str) {
            if (c == '<') {
                depth++;
                if (depth > maxDepth) {
                    maxDepth = depth;
                }
            } else if (c == '>') {
                depth--;
            }
        }

        return maxDepth;
    }

    static std::string FormatParam(const std::string& param, int level, const FormatConfig& config, int& lineCount)
    {
        size_t templateStart = param.find('<');
        if (templateStart == std::string::npos) {
            return param;
        }

        return FormatTemplateInternal(param, level, config);
    }

    static bool IsAllSample(const std::vector<std::string>& params)
    {
        bool allSimple = true;
        for (const auto& p : params) {
            if (p.find('<') != std::string::npos) {
                allSimple = false;
                break;
            }
        }
        return allSimple;
    }

    static std::string GetSingleLineStr(const std::vector<std::string>& params, const std::string& baseType)
    {
        std::ostringstream singleLine;
        singleLine << baseType << "<";
        for (size_t i = 0; i < params.size(); i++) {
            singleLine << params[i];
            if (i != params.size() - 1) {
                singleLine << ", ";
            }
        }
        singleLine << ">";
        return singleLine.str();
    }

    static std::string FormatTemplateInternal(const std::string& typeName, int level, const FormatConfig& config)
    {
        auto templateStart = typeName.find('<');
        if (templateStart == std::string::npos) {
            return typeName;
        }

        auto baseType = typeName.substr(0, templateStart);

        if (templateStart + 1 >= typeName.length() || typeName[templateStart + 1] == '>') {
            return baseType + "<>";
        }

        auto paramStr = typeName.substr(templateStart + 1);
        if (!paramStr.empty() && paramStr.back() == '>') {
            paramStr.pop_back();
        }

        auto params = SplitTemplateParams(paramStr);
        std::string singleLineStr = GetSingleLineStr(params, baseType);
        auto currentLineLength = CreateIndent(level, config).length() + singleLineStr.length();
        bool needsMultiline = NeedsMultiLine(typeName, config, params, currentLineLength);
        if (!needsMultiline) {
            return singleLineStr;
        }

        return GetMultiLineStr(level, config, baseType, params);
    }

    static std::string GetMultiLineStr(
        int level, const FormatConfig& config, const std::string& baseType, const std::vector<std::string>& params)
    {
        std::ostringstream result;
        result << baseType << "<";
        bool allSimple = IsAllSample(params);
        if (allSimple && params.size() <= config.maxParamsPerLine) {
            result << "\n" << CreateIndent(level + 1, config);
            for (size_t i = 0; i < params.size(); i++) {
                result << params[i];
                if (i != params.size() - 1) {
                    result << ", ";
                }
            }

            result << "\n" << CreateIndent(level, config) << ">";
            return result.str();
        }

        result << "\n";
        for (size_t i = 0; i < params.size(); i++) {
            int paramLineCount = 0;
            std::string formattedParam = FormatParam(params[i], level + 1, config, paramLineCount);
            result << CreateIndent(level + 1, config) << formattedParam;
            if (i != params.size() - 1) {
                result << ",";
            }
            result << "\n";
        }

        result << CreateIndent(level, config) << ">";
        return result.str();
    }

    static bool NeedsMultiLine(
        const std::string& typeName, const FormatConfig& config, const std::vector<std::string>& params,
        size_t currentLineLength)
    {
        bool needsMultiline = false;

        if (currentLineLength > config.maxLineLength) {
            needsMultiline = true;
        }

        if (params.size() > config.maxParamsPerLine) {
            needsMultiline = true;
        }

        for (const auto& p : params) {
            if (IsComplexTemplate(p)) {
                needsMultiline = true;
                break;
            }
        }

        if (GetTemplateDepth(typeName) > config.maxTemplateDepthPerLine) {
            needsMultiline = true;
        }
        return needsMultiline;
    }

public:
    static std::string Pretty(const std::string& typeName)
    {
        FormatConfig config;
        config.indentSize = 4;
        config.maxLineLength = 120;
        config.maxParamsPerLine = 3;
        config.maxTemplateDepthPerLine = 2;
        return FormatTemplateInternal(typeName, 0, config);
    }
};

template <typename T>
std::string GetPrettyTypeName()
{
    return TemplatePrettyFormater::Pretty(GetTypeName<T>());
}

template <typename T>
std::string GetPrettyTypeName(const T& value)
{
    return TemplatePrettyFormater::Pretty(GetTypeName<T>());
}

#define PRINT_TYPE(T) GetTypeName<T>().c_str()

#define RUN_LOG_ONE_BLOCK(...)                                                   \
    do {                                                                         \
        const char* filename = strrchr(__FILE__, '/');                           \
        if (!filename)                                                           \
            filename = strrchr(__FILE__, '\\');                                  \
        filename = filename ? filename + 1 : __FILE__;                           \
        if constexpr (sizeof(#__VA_ARGS__) <= 1) {                               \
            printf("[INFO][Core0:%s:%d] (empty log)\n", filename, __LINE__);     \
        } else {                                                                 \
            char buffer[1024];                                                   \
            snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, __VA_ARGS__); \
            if (buffer[strlen(buffer) - 1] != '\n') {                            \
                printf("[INFO][Core0:%s:%d] %s\n", filename, __LINE__, buffer);  \
            } else {                                                             \
                printf("[INFO][Core0:%s:%d] %s", filename, __LINE__, buffer);    \
            }                                                                    \
        }                                                                        \
    } while (0)

/////////////////////////////////////////////////////////////////////////////

#ifdef __CCE_KT_TEST__
// CPU孪生调试时支持打印
#define RUN_LOG(...)                    \
    if (GetBlockIdx() == 0) {           \
        RUN_LOG_ONE_BLOCK(__VA_ARGS__); \
    }

template <typename T, T v>
struct Print2 {
    constexpr operator char()
    {
        return 1 + 0xFF;
    }
};
#define BUILD_LOG(...) char UNIQUE_NAME(print_value_) = Print2<__VA_ARGS__>()

#else // __CCE_KT_TEST__
#ifdef __ATP_UT__
#define RUN_LOG(...) RUN_LOG_ONE_BLOCK(__VA_ARGS__)
#else // __ATP_UT__
// 实际编译Kernel时不打印
#define RUN_LOG(...)
#endif // __ATP_UT__
#define BUILD_LOG(...)
#endif // __CCE_KT_TEST__

#endif // ATVOSS_LOG_H_
