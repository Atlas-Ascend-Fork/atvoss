/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <cstdlib>
#include <algorithm>

class CommandLine {
private:
    std::map<std::string, std::string> args_;

    template <typename T>
    T FromString(const std::string& str) const;

public:
    CommandLine(int argc, char const* argv[]);

    template <typename T>
    T Get(const std::string& name, const T& default_val = T{}) const;

    bool Present(const std::string& name) const;
};

// String → T 转换
template <typename T>
T CommandLine::FromString(const std::string& str) const
{
    std::stringstream ss(str);
    T value;
    ss >> value;
    return value;
}

template <>
inline int CommandLine::FromString<int>(const std::string& str) const
{
    return std::atoi(str.c_str());
}
template <>
inline float CommandLine::FromString<float>(const std::string& str) const
{
    return std::atof(str.c_str());
}
template <>
inline double CommandLine::FromString<double>(const std::string& str) const
{
    return std::atof(str.c_str());
}
template <>
inline bool CommandLine::FromString<bool>(const std::string& str) const
{
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s == "true" || s == "1" || s == "on" || s == "yes";
}
template <>
inline std::string CommandLine::FromString<std::string>(const std::string& str) const
{
    return str;
}

template <>
inline std::vector<int> CommandLine::FromString<std::vector<int>>(const std::string& str) const
{
    std::vector<int> result;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (!token.empty()) {
            result.push_back(std::atoi(token.c_str()));
        }
    }
    return result;
}

CommandLine::CommandLine(int argc, char const* argv[])
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.substr(0, 2) == "--") {
            size_t eq_pos = arg.find('=');
            std::string key = (eq_pos == std::string::npos) ? arg.substr(2) : arg.substr(2, eq_pos - 2);
            std::string val;
            if (eq_pos != std::string::npos) {
                val = arg.substr(eq_pos + 1);
            } else {
                val = (i + 1 < argc && argv[i + 1][0] != '-') ? argv[++i] : "true";
            }
            args_[key] = val;
        }
    }
}

template <typename T>
T CommandLine::Get(const std::string& name, const T& default_val) const
{
    auto it = args_.find(name);
    if (it == args_.end())
        return default_val;
    try {
        return FromString<T>(it->second);
    } catch (...) {
        return default_val;
    }
}

bool CommandLine::Present(const std::string& name) const
{
    return args_.find(name) != args_.end();
}