/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXPR_H 
#define EXPR_H 
#include <iostream>
#include <tuple>
#include <cstdint>
#include <typeinfo>
#include <cxxabi.h>

#pragma region "argslist"
//完成任意类型的数据存储
template<class... Args>
struct ArgList{};
 
template<class T>
struct ArgList<T>{
    T value;
    template<int pos>
    auto Get() { return value; }
};
template<class T, class... Args>
struct ArgList<T, Args...> : public ArgList<Args...> {    
    T value;
    template<int pos>
    auto Get() {
        if constexpr(pos == 0) {
            return value;
        }
        else {
            return ArgList<Args...>::template Get<pos-1>();
        }
    }
};
#pragma endregion "argslist"

#pragma region "expression"
 
template<typename T, typename Op>
struct _ReplaceExprDst {    
};
template<class T, template<class...> class Op, class R, class ... Args>
struct _ReplaceExprDst<T, Op<R, Args...> > {
    using Type = Op<T, Args...>;    
};
 
template<typename T>
struct Expr  {
    using Op = T;
    T data;
    template<typename U>
    auto operator = (const Expr<U>& op) {
        using OpCode = _ReplaceExprDst<T, U>;
        return Expr< typename OpCode::Type > { { data, op.data.src } }; 
    }
};
#pragma endregion "expression"
 
#pragma region "param"
template<int pos, typename T, typename Layout_ = void, class argsType=void>
struct Param {
    constexpr static int placeHoder = pos;
    using ArgsType = argsType;
    using DataType = T; 
    using Layout = Layout_;
    T dst;
};

namespace ParamType { 
    struct In{};
    struct Out{};
    struct InOut{};
    struct Temp{};
    struct PreOut{};
    struct PostIn{};
};

#pragma endregion "param"

template<class Op>
struct GetExprParam {};

template<template<class...> class Op, class Dst>
struct GetExprParam< Expr< Op<Dst> >> {
    using Type = std::tuple< Dst>;  //检查dst 是否为 ParamType
};

template<template<class...> class Op, class Dst, class Src1>
struct GetExprParam< Expr< Op<Dst, Src1> >> {
    using Type = std::tuple< Dst, Src1>;  //检查dst 是否为 ParamType
};

template<template<class...> class Op, class Dst, class Src1, class Src2>
struct GetExprParam< Expr< Op<Dst, Src1, Src2> >> {
    using Type = std::tuple< Dst, Src1, Src2>;  //检查dst 是否为 ParamType
};
template<template<class...> class Op, class Dst, class Src1, class Src2, class Src3>
struct GetExprParam< Expr< Op<Dst, Src1, Src2, Src3> >> {
    using Type = std::tuple< Dst, Src1, Src2, Src3>;  //检查dst 是否为 ParamType
};

template<class Op, class T>
struct GetParamList {
    using Type = void;
};

template<class... Op, class T>
struct GetParamList<std::tuple<Op...>, T> { 
    template<class Head>
    struct ParamCheck {
        constexpr static bool value =  std::is_same_v<typename Head::ArgsType, T>;
    }; 

    using Params = Util::Concat_t< typename GetExprParam<Op>::Type ... >;
    using Union = Util::Unique_t<Params>;
    using Type = Util::Filter_t<ParamCheck, Union>;
};

//定义参数输入输出类型
template<class T, typename Layout = void>
struct In : public Expr<Param<-1, T, Layout, ParamType::In >> {    
    using Parent = Expr<Param<-1, T, Layout, ParamType::In >>;
    using Parent::operator=;
};
template<class T, typename Layout = void>
struct Out : public Expr<Param<-1, T, Layout, ParamType::Out >> {    
    using Parent = Expr<Param<-1, T, Layout, ParamType::Out >>;
    using Parent::operator=;
};
 
template<class ... Args>
struct Ins {
    using Type = std::tuple<Args...>;
};

template<class ... Args>
struct Outs {
    using Type = std::tuple<Args...>;
};

template<class ... Args>
struct InOuts {
    using Type = std::tuple<Args...>;
};

template<class ... Args>
struct Temps {
    using Type = std::tuple<Args...>;
};

template<int holder, typename T, typename Layout=void>
auto PlaceHoderIn() {
    return Expr<Param<holder, T, Layout, ParamType::In>>{ };
}
template<int holder, typename T, typename Layout=void>
auto PlaceHoderOut() {
    return Expr<Param<holder, T, Layout, ParamType::Out>>{ };
}

template<class T>
struct TypeAux {
    using Type = T;
};
template<>
struct TypeAux<void> {
    using Type = struct {};
};

template<class R, class T, class U>
struct OpAdd {
    typename TypeAux<R>::Type dst;
    ArgList<T, U> src;
    auto Run() {
        return  src.template Get<0>().dst + src.template Get<1>().dst;
    }
    template<class Ths, class Uhs>
    void Op(Ths& ths, Uhs& uhs){
    }
};

template<class T, class U>
auto operator + (Expr<T>& t, Expr<U>& u) {
    return Expr< OpAdd<void, T, U> >{};
}


#endif //EXPR_H 