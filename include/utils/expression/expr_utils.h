/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef Atvoss_EXPR_UTILS_H
#define Atvoss_EXPR_UTILS_H

namespace Util {
template <typename T>
struct TypeWrapper {
    using Type = T;
};

template <typename List>
struct First{};

template <typename Head, typename... Tail>
struct First<std::tuple<Head, Tail...>> {
    using Type = Head;
};
template <typename List>
using First_t = typename First<List>::Type;

template <typename List>
struct Size{};

template <typename... Ts>
struct Size<std::tuple<Ts...>> {
    static constexpr int value = sizeof...(Ts);
};
template <typename List>
inline constexpr auto Size_v = Size<List>::value;

template <typename T, typename List>
struct Prepend{};

template <typename T, typename... Ts>
struct Prepend<T, std::tuple<Ts...>> {
    using Type = std::tuple<T, Ts...>;
};

template <typename T, typename List>
using Prepend_t = typename Prepend<T, List>::Type;

template <typename... Lists>
struct Concat{};

template <>
struct Concat<> {
    using Type = std::tuple<>;
};

template <typename... Ts>
struct Concat<std::tuple<Ts...>> {
    using Type = std::tuple<Ts...>;
};

template <typename... Ts, typename... Us>
struct Concat<std::tuple<Ts...>, std::tuple<Us...>> {
    using Type = std::tuple<Ts..., Us...>;
};

template <typename List1, typename List2,  typename... Lists>
struct Concat<List1, List2,  Lists...> {
    using Type = typename Concat<typename Concat<List1, List2>::Type,  Lists...>::Type;
};

template <typename... Lists>
using Concat_t = typename Concat<Lists...>::Type;

template <template <typename> class Pred, typename List>
struct Check{};

template <template <typename> class Pred, typename Head, typename... Tail>
struct Check<Pred, std::tuple<Head, Tail...>> {
    static constexpr bool value = Pred<Head>::value && Check<Pred, std::tuple<Tail...>>::value;
};

template <template <typename> class Pred>
struct Check<Pred, std::tuple<>> {
    static constexpr bool value = true;
};

template <template <typename> class Pred, typename List>
inline constexpr bool Check_v = Check<Pred, List>::value;

template <template <typename> class Pred, typename List>
struct Filter{};

template <template <typename> class Pred, typename Head, typename... Tail>
struct Filter<Pred, std::tuple<Head, Tail...>> {
    using Type = std::conditional_t<Pred<Head>::value, 
            Prepend_t<Head, typename Filter<Pred, std::tuple<Tail...>>::Type>,
            typename Filter<Pred, std::tuple<Tail...>>::Type>;
};

template <template <typename> class Pred>
struct Filter<Pred, std::tuple<>> {
    using Type = std::tuple<>;
};

template <template <typename> class Pred, typename List>
using Filter_t = typename Filter<Pred, List>::Type;

template <template <typename> class Pred, typename List, typename = void>
struct Find{};

template <template <typename> class Pred>
struct Find<Pred, std::tuple<>> {
    static constexpr int value = 0;
};

template <template <typename> class Pred, typename Head, typename... Tail>
struct Find<Pred, std::tuple<Head, Tail...>, std::enable_if_t<Pred<Head>::value>> {
    static constexpr int value = 0;
};

template <template <typename> class Pred, typename Head, typename... Tail>
struct Find<Pred, std::tuple<Head, Tail...>, std::enable_if_t<!Pred<Head>::value>> {
    static constexpr int value = Find<Pred, std::tuple<Tail...>>::value + 1;
};

template <template <typename> class Pred, typename List>
inline constexpr int Find_v = Find<Pred, List>::value;

template <typename List, int N>
struct Get{};

template <typename Head, typename... Tail, int N>
struct Get<std::tuple<Head, Tail...>, N> {
    using Type = typename Get<std::tuple<Tail...>, N - 1>::Type;
};

template <typename Head, typename... Tail>
struct Get<std::tuple<Head, Tail...>, 0> {
    using Type = Head;
};

template <int N>
struct Get<std::tuple<>, N> {
    static_assert(N < 0, "[ERROR]: [Atvoss][Expression] Index out of bounds in Get");
};

template <typename List, int N>
using Get_t = typename Get<List, N>::Type;

template <typename List>
struct Unique{};

template <typename Head, typename... Tail>
struct Unique<std::tuple<Head, Tail...>> {
    template <typename T>
    struct IsNotHead : std::bool_constant<!std::is_same_v<T, Head>> {
    };

    using Type = Prepend_t<Head, typename Unique<Filter_t<IsNotHead, std::tuple<Tail...>>>::Type>;
};

template <>
struct Unique<std::tuple<>> {
    using Type = std::tuple<>;
};

template <typename List>
using Unique_t = typename Unique<List>::Type;

template <template <typename> class Pred, typename List>
struct FindUnique {
    using ResultList = Filter_t<Pred, List>;
    static_assert(Size_v<ResultList> != 0, "[ERROR]: [Atvoss][Expression] Cannot find the specified element");
    static_assert(Size_v<ResultList> == 1, "[ERROR]: [Atvoss][Expression] A unique result is expected");
    using Type = First_t<ResultList>;
};

template <template <typename> class Pred, typename List>
using FindUnique_t = typename FindUnique<Pred, List>::Type;

template <typename List, typename T>
struct Contains{};

template <typename T>
struct Contains<std::tuple<>, T> : std::false_type {
};

template <typename... Tail, typename T>
struct Contains<std::tuple<T, Tail...>, T> : std::true_type {
};

template <typename Head, typename... Tail, typename T>
struct Contains<std::tuple<Head, Tail...>, T> {
    static constexpr bool value = Contains<std::tuple<Tail...>, T>::value;
};

template <typename List, typename T>
inline constexpr bool Contains_v = Contains<List, T>::value;

template <typename List1, typename List2>
struct Intersection {
    template <typename T>
    struct IsInList2 : Contains<List2, T> {
    };

    using Type = Filter_t<IsInList2, List1>;
};

template <typename List1, typename List2>
using Intersection_t = typename Intersection<List1, List2>::Type;

template <typename List1, typename List2>
struct Difference {
    template <typename T>
    struct IsNotInList2 {
        static constexpr bool value = !Contains_v<List2, T>;
    };

    using Type = Filter_t<IsNotInList2, List1>;
};

template <typename List1, typename List2>
using Difference_t = typename Difference<List1, List2>::Type;

template <typename Func, typename Data>
auto ForEach(std::tuple<>, Func&& /*func*/, Data data)
{
    return data;
}

template <typename Head, typename... Tail, typename Func, typename Data>
auto ForEach(std::tuple<Head, Tail...>, Func&& func, Data data)
{
    return ForEach(std::tuple<Tail...>{}, std::forward<Func>(func), std::forward<Func>(func)(TypeWrapper<Head>{}, data));
} 
 
}  // namespace Util

#endif // EXPR_UTILS_H