/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATVOSS_UTILITY_H
#define ATVOSS_UTILITY_H

namespace Atvoss::Util {

template <typename T>
struct AlwaysFalse : std::false_type {};

template <typename T>
inline constexpr bool AlwaysFalse_v = AlwaysFalse<T>::value;

template <template <typename...> class T, typename U>
struct IsSpecializationOf : std::false_type {};

template <template <typename...> class T, typename... Us>
struct IsSpecializationOf<T, T<Us...>> : std::true_type {};

template <template <typename...> class T, typename U>
inline constexpr bool IsSpecializationOf_v = IsSpecializationOf<T, U>::value;

template <typename T, typename = void>
struct HasClear : std::false_type {};

template <typename T>
struct HasClear<T, std::void_t<decltype(std::declval<T&>().Clear())>> : std::true_type {};

template <typename T>
inline constexpr bool HasClear_v = HasClear<T>::value;

template <
    typename Tuple, typename Func, std::enable_if_t<IsSpecializationOf_v<std::tuple, std::decay_t<Tuple>>, int> = 0>
void ForEach(Tuple&& tuple, Func&& func)
{
    std::apply(
        [&func](auto&&... args) { (std::forward<Func>(func)(std::forward<decltype(args)>(args)), ...); },
        std::forward<Tuple>(tuple));
}
namespace Detail {

// Check if compression would cause ambiguity
template <typename T1, typename T2>
inline constexpr bool WouldCauseAmbiguity_v =
    std::is_same_v<T1, T2> || std::is_base_of_v<T1, T2> || std::is_base_of_v<T2, T1>;

// Helper to determine if a type can be compressed
template <typename T>
inline constexpr bool CanCompress_v = std::is_empty_v<T> && !std::is_final_v<T>;

// Helpers to determine compression flags, accounting for ambiguity
template <typename T1, typename T2>
inline constexpr bool CompressFirst_v = CanCompress_v<T1> && (!CanCompress_v<T2> || !WouldCauseAmbiguity_v<T1, T2>);
template <typename T1, typename T2>
inline constexpr bool CompressSecond_v = CanCompress_v<T2> && (!CanCompress_v<T1> || !WouldCauseAmbiguity_v<T1, T2>);

// Storage implementations for different compression scenarios
template <typename T, bool Compress>
struct CompressedDataStorage;

// Case 1: Not compressed
template <typename T>
struct CompressedDataStorage<T, false> {
    template <typename U, typename = std::enable_if_t<!std::is_same_v<std::decay_t<U>, CompressedDataStorage>>>
    explicit constexpr CompressedDataStorage(U&& u) : data_(std::forward<U>(u))
    {}

    CompressedDataStorage() = default;

    T& Data()
    {
        return data_;
    }

    const T& Data() const
    {
        return data_;
    }

private:
    T data_;
};

// Case 2: Compressed
template <typename T>
struct CompressedDataStorage<T, true> : private T {
    template <typename U, typename = std::enable_if_t<!std::is_same_v<std::decay_t<U>, CompressedDataStorage>>>
    explicit constexpr CompressedDataStorage(U&& u) : T(std::forward<U>(u))
    {}

    CompressedDataStorage() = default;

    T& Data()
    {
        return static_cast<T&>(*this);
    }

    const T& Data() const
    {
        return static_cast<const T&>(*this);
    }
};

// Storage implementations for different compression scenarios
template <typename T1, typename T2, bool CompressFirst, bool CompressSecond>
struct CompressedPairStorage;

// Case 1: Neither can be compressed
template <typename T1, typename T2>
struct CompressedPairStorage<T1, T2, false, false> {
    template <typename U1, typename U2>
    explicit constexpr CompressedPairStorage(U1&& f, U2&& s) : first_(std::forward<U1>(f)), second_(std::forward<U2>(s))
    {}

    CompressedPairStorage() = default;

    T1& First()
    {
        return first_;
    }

    const T1& First() const
    {
        return first_;
    }

    T2& Second()
    {
        return second_;
    }

    const T2& Second() const
    {
        return second_;
    }

private:
    T1 first_;
    T2 second_;
};

// Case 2: Only first can be compressed
template <typename T1, typename T2>
struct CompressedPairStorage<T1, T2, true, false> : private T1 {
    template <typename U1, typename U2>
    explicit constexpr CompressedPairStorage(U1&& f, U2&& s) : T1(std::forward<U1>(f)), second_(std::forward<U2>(s))
    {}

    CompressedPairStorage() = default;

    T1& First()
    {
        return static_cast<T1&>(*this);
    }

    const T1& First() const
    {
        return static_cast<const T1&>(*this);
    }

    T2& Second()
    {
        return second_;
    }

    const T2& Second() const
    {
        return second_;
    }

private:
    T2 second_;
};

// Case 3: Only second can be compressed
template <typename T1, typename T2>
struct CompressedPairStorage<T1, T2, false, true> : private T2 {
    template <typename U1, typename U2>
    explicit constexpr CompressedPairStorage(U1&& f, U2&& s) : T2(std::forward<U2>(s)), first_(std::forward<U1>(f))
    {}

    CompressedPairStorage() = default;

    T1& First()
    {
        return first_;
    }

    const T1& First() const
    {
        return first_;
    }

    T2& Second()
    {
        return static_cast<T2&>(*this);
    }

    const T2& Second() const
    {
        return static_cast<const T2&>(*this);
    }

private:
    T1 first_;
};

// Case 4: Both can be compressed, different types
template <typename T1, typename T2>
struct CompressedPairStorage<T1, T2, true, true> : private T1, private T2 {
    template <typename U1, typename U2>
    explicit constexpr CompressedPairStorage(U1&& f, U2&& s) : T1(std::forward<U1>(f)), T2(std::forward<U2>(s))
    {}

    CompressedPairStorage() = default;

    T1& First()
    {
        return static_cast<T1&>(*this);
    }

    const T1& First() const
    {
        return static_cast<const T1&>(*this);
    }

    T2& Second()
    {
        return static_cast<T2&>(*this);
    }

    const T2& Second() const
    {
        return static_cast<const T2&>(*this);
    }
};

// Case 5: Both can be compressed, same type
template <typename T>
struct CompressedPairStorage<T, T, true, true> : private T {
    template <typename U1, typename U2>
    explicit constexpr CompressedPairStorage(U1&& f, U2&&) : T(std::forward<U1>(f))
    {}

    CompressedPairStorage() = default;

    T& First()
    {
        return static_cast<T&>(*this);
    }

    const T& First() const
    {
        return static_cast<const T&>(*this);
    }

    T& Second()
    {
        return static_cast<T&>(*this);
    }

    const T& Second() const
    {
        return static_cast<const T&>(*this);
    }
};

} // namespace Detail

template <typename T>
class CompressedData : private Detail::CompressedDataStorage<T, std::is_empty_v<T> && !std::is_final_v<T>> {
    using Storage = Detail::CompressedDataStorage<T, std::is_empty_v<T> && !std::is_final_v<T>>;

public:
    using Type = T;

    // Default constructor
    CompressedData() = default;

    // Perfect forwarding constructor
    template <typename U, typename = std::enable_if_t<!std::is_same_v<std::decay_t<U>, CompressedData>>>
    explicit constexpr CompressedData(U&& u) : Storage(std::forward<U>(u))
    {}

    // Accessors for data
    T& Data()
    {
        return Storage::Data();
    }

    const T& Data() const
    {
        return Storage::Data();
    }

    // Swap
    void swap(CompressedData& other) noexcept
    {
        using std::swap;
        swap(Data(), other.Data());
    }
};

// Swap function
template <typename T>
void swap(CompressedData<T>& lhs, CompressedData<T>& rhs) noexcept
{
    lhs.swap(rhs);
}

// Equality comparison operators
template <typename T>
bool operator==(const CompressedData<T>& lhs, const CompressedData<T>& rhs)
{
    return lhs.Data() == rhs.Data();
}

template <typename T>
bool operator!=(const CompressedData<T>& lhs, const CompressedData<T>& rhs)
{
    return !(lhs == rhs);
}

template <typename T1, typename T2>
class CompressedPair
    : private Detail::CompressedPairStorage<T1, T2, Detail::CompressFirst_v<T1, T2>, Detail::CompressSecond_v<T1, T2>> {
    using Storage =
        Detail::CompressedPairStorage<T1, T2, Detail::CompressFirst_v<T1, T2>, Detail::CompressSecond_v<T1, T2>>;

public:
    using FirstType = T1;
    using SecondType = T2;

    // Default constructor
    CompressedPair() = default;

    // Perfect forwarding constructor
    template <typename U1, typename U2>
    explicit constexpr CompressedPair(U1&& f, U2&& s) : Storage(std::forward<U1>(f), std::forward<U2>(s))
    {}

    // Accessors for first element
    T1& First()
    {
        return Storage::First();
    }

    const T1& First() const
    {
        return Storage::First();
    }

    // Accessors for second element
    T2& Second()
    {
        return Storage::Second();
    }

    const T2& Second() const
    {
        return Storage::Second();
    }

    // Swap
    void swap(CompressedPair& other) noexcept
    {
        using std::swap;
        swap(First(), other.First());
        swap(Second(), other.Second());
    }
};

// Deduction guide
template <typename T1, typename T2>
CompressedPair(T1&& f, T2&& s) -> CompressedPair<std::decay_t<T1>, std::decay_t<T2>>;

// Swap function
template <typename T1, typename T2>
void swap(CompressedPair<T1, T2>& lhs, CompressedPair<T1, T2>& rhs) noexcept
{
    lhs.swap(rhs);
}

// Equality comparison operators
template <typename T1, typename T2>
bool operator==(const CompressedPair<T1, T2>& lhs, const CompressedPair<T1, T2>& rhs)
{
    return lhs.First() == rhs.First() && lhs.Second() == rhs.Second();
}

template <typename T1, typename T2>
bool operator!=(const CompressedPair<T1, T2>& lhs, const CompressedPair<T1, T2>& rhs)
{
    return !(lhs == rhs);
}

template <typename T>
struct TypeWrapper {
    using Type = T;
};

template <typename T>
struct RetTypeWrapper {
    using RetType = T;
};

template <typename T>
struct TensorTypeWrapper {
    using TensorType = T;
};

template <typename... Ts>
struct TypeList {};

template <typename T>
struct IsTypeList : std::false_type {};

template <typename... Ts>
struct IsTypeList<TypeList<Ts...>> : std::true_type {};

template <typename T>
inline constexpr bool IsTypeList_v = IsTypeList<T>::value;

template <typename List>
struct First;

template <>
struct First<TypeList<>> {
    using Type = void;
};

template <typename Head, typename... Tail>
struct First<TypeList<Head, Tail...>> {
    using Type = Head;
};

template <typename List>
using First_t = typename First<List>::Type;

template <typename List>
struct Size;

template <typename... Ts>
struct Size<TypeList<Ts...>> {
    static constexpr std::size_t value = sizeof...(Ts);
};

template <typename List>
inline constexpr std::size_t Size_v = Size<List>::value;

template <typename List, typename T>
struct Append;

template <typename... Ts, typename T>
struct Append<TypeList<Ts...>, T> {
    using Type = TypeList<Ts..., T>;
};

template <typename List, typename T>
using Append_t = typename Append<List, T>::Type;

template <typename T, typename List>
struct Prepend;

template <typename T, typename... Ts>
struct Prepend<T, TypeList<Ts...>> {
    using Type = TypeList<T, Ts...>;
};

template <typename T, typename List>
using Prepend_t = typename Prepend<T, List>::Type;

template <typename... Lists>
struct Concatenate;

template <>
struct Concatenate<> {
    using Type = TypeList<>;
};

template <typename... Ts>
struct Concatenate<TypeList<Ts...>> {
    using Type = TypeList<Ts...>;
};

template <typename... Ts, typename... Us>
struct Concatenate<TypeList<Ts...>, TypeList<Us...>> {
    using Type = TypeList<Ts..., Us...>;
};

template <typename List1, typename List2, typename List3, typename... Lists>
struct Concatenate<List1, List2, List3, Lists...> {
    using Type = typename Concatenate<typename Concatenate<List1, List2>::Type, List3, Lists...>::Type;
};

template <typename... Lists>
using Concatenate_t = typename Concatenate<Lists...>::Type;

template <template <typename> class Pred, typename List>
struct All;

template <template <typename> class Pred, typename Head, typename... Tail>
struct All<Pred, TypeList<Head, Tail...>>
    : std::conditional_t<!Pred<Head>::value, std::false_type, All<Pred, TypeList<Tail...>>> {};

template <template <typename> class Pred>
struct All<Pred, TypeList<>> : std::true_type {};

template <template <typename> class Pred, typename List>
inline constexpr bool All_v = All<Pred, List>::value;

template <template <typename> class Pred, typename List>
struct Any;

template <template <typename> class Pred, typename Head, typename... Tail>
struct Any<Pred, TypeList<Head, Tail...>>
    : std::conditional_t<Pred<Head>::value, std::true_type, Any<Pred, TypeList<Tail...>>> {};

template <template <typename> class Pred>
struct Any<Pred, TypeList<>> : std::false_type {};

template <template <typename> class Pred, typename List>
inline constexpr bool Any_v = Any<Pred, List>::value;

template <template <typename> class Pred, typename List>
struct Filter;

template <template <typename> class Pred, typename Head, typename... Tail>
struct Filter<Pred, TypeList<Head, Tail...>> {
    using Type = typename std::conditional_t<
        Pred<Head>::value, Prepend_t<Head, typename Filter<Pred, TypeList<Tail...>>::Type>,
        typename Filter<Pred, TypeList<Tail...>>::Type>;
};

template <template <typename> class Pred>
struct Filter<Pred, TypeList<>> {
    using Type = TypeList<>;
};

template <template <typename> class Pred, typename List>
using Filter_t = typename Filter<Pred, List>::Type;

template <template <typename> class Pred, typename List, typename = void>
struct Find;

template <template <typename> class Pred>
struct Find<Pred, TypeList<>> {
    static constexpr std::size_t value = 0;
};

template <template <typename> class Pred, typename Head, typename... Tail>
struct Find<Pred, TypeList<Head, Tail...>, std::enable_if_t<Pred<Head>::value>> {
    static constexpr std::size_t value = 0;
};

template <template <typename> class Pred, typename Head, typename... Tail>
struct Find<Pred, TypeList<Head, Tail...>, std::enable_if_t<!Pred<Head>::value>> {
    static constexpr std::size_t value = Find<Pred, TypeList<Tail...>>::value + 1;
};

template <template <typename> class Pred, typename List>
inline constexpr std::size_t Find_v = Find<Pred, List>::value;

namespace Detail {

template <template <typename> class Pred, typename List, std::size_t Current, std::size_t Last>
struct FindLastImpl;

template <template <typename> class Pred, std::size_t Current, std::size_t Last>
struct FindLastImpl<Pred, TypeList<>, Current, Last> {
    static constexpr std::size_t value = (Last == static_cast<std::size_t>(-1)) ? Current : Last;
};

template <template <typename> class Pred, typename Head, typename... Tail, std::size_t Current, std::size_t Last>
struct FindLastImpl<Pred, TypeList<Head, Tail...>, Current, Last> {
    static constexpr std::size_t value = std::conditional_t<
        Pred<Head>::value, FindLastImpl<Pred, TypeList<Tail...>, Current + 1, Current>,
        FindLastImpl<Pred, TypeList<Tail...>, Current + 1, Last>>::value;
};

} // namespace Detail

template <template <typename> class Pred, typename List>
struct FindLast : Detail::FindLastImpl<Pred, List, 0, static_cast<std::size_t>(-1)> {};

template <template <typename> class Pred, typename List>
inline constexpr std::size_t FindLast_v = FindLast<Pred, List>::value;

template <typename List, std::size_t N>
struct Get;

template <typename Head, typename... Tail, std::size_t N>
struct Get<TypeList<Head, Tail...>, N> {
    using Type = typename Get<TypeList<Tail...>, N - 1>::Type;
};

template <typename Head, typename... Tail>
struct Get<TypeList<Head, Tail...>, 0> {
    using Type = Head;
};

template <std::size_t N>
struct Get<TypeList<>, N> {
    static_assert(N < 0, "[ERROR]: [Atvoss][Expression] Index out of bounds in Get");
};

template <typename List, std::size_t N>
using Get_t = typename Get<List, N>::Type;

template <typename List, std::size_t N>
struct GetFirstN;

template <typename Head, typename... Tail, std::size_t N>
struct GetFirstN<TypeList<Head, Tail...>, N> {
    using Type = Concatenate_t<TypeList<Head>, typename GetFirstN<TypeList<Tail...>, N - 1>::Type>;
};

template <typename List>
struct GetFirstN<List, 0> {
    using Type = TypeList<>;
};

template <typename List, std::size_t N>
using GetFirstN_t = typename GetFirstN<List, N>::Type;

template <typename List, std::size_t N, typename = void>
struct RemoveFirstN {
    using Type = List;
};

template <typename Head, typename... Tail, std::size_t N>
struct RemoveFirstN<TypeList<Head, Tail...>, N, std::enable_if_t<(N > 0)>> {
    using Type = typename RemoveFirstN<TypeList<Tail...>, N - 1>::Type;
};

template <typename List, std::size_t N>
using RemoveFirstN_t = typename RemoveFirstN<List, N>::Type;

template <typename List, std::size_t N>
struct GetLastN {
private:
    constexpr static std::size_t total = Size_v<List>;
    constexpr static std::size_t skip = (N < 0) ? total : ((N < total) ? total - N : 0);

public:
    using Type = typename RemoveFirstN<List, skip>::Type;
};

template <typename List, std::size_t N>
using GetLastN_t = typename GetLastN<List, N>::Type;

template <typename List>
struct IsEmpty : std::false_type {};

template <>
struct IsEmpty<TypeList<>> : std::true_type {};

template <typename List>
inline constexpr bool IsEmpty_v = IsEmpty<List>::value;

template <typename List>
struct IsNotEmpty : std::true_type {};

template <>
struct IsNotEmpty<TypeList<>> : std::false_type {};

template <typename List>
inline constexpr bool IsNotEmpty_v = IsNotEmpty<List>::value;

template <typename List>
struct Rest;

template <typename Head, typename... Tail>
struct Rest<TypeList<Head, Tail...>> {
    using Type = TypeList<Tail...>;
};

template <typename List>
using Rest_t = typename Rest<List>::Type;

template <typename List, std::size_t N>
struct Drop {
    static_assert(IsNotEmpty_v<List>, "Cannot drop from empty list");
    using Type = typename Drop<Rest_t<List>, N - 1>::Type;
};

template <typename List>
struct Drop<List, 0> {
    using Type = List;
};

template <typename List, std::size_t N>
using Drop_t = typename Drop<List, N>::Type;

template <template <typename...> class Proc, typename List>
struct Apply;

template <template <typename...> class Proc, typename... Args>
struct Apply<Proc, TypeList<Args...>> {
    using Type = typename Proc<Args...>::Type;
};

template <template <typename...> class Proc, typename List>
using Apply_t = typename Apply<Proc, List>::Type;

template <template <typename> class Proc, typename List>
struct Map;

template <template <typename> class Proc, typename... Ts>
struct Map<Proc, TypeList<Ts...>> {
    using Type = TypeList<typename Proc<Ts>::Type...>;
};

template <template <typename> class Proc, typename List>
using Map_t = typename Map<Proc, List>::Type;

template <typename List>
struct Reverse;

template <>
struct Reverse<TypeList<>> {
    using Type = TypeList<>;
};

template <typename Head, typename... Tail>
struct Reverse<TypeList<Head, Tail...>> {
    using Type = Append_t<typename Reverse<TypeList<Tail...>>::Type, Head>;
};

template <typename List>
using Reverse_t = typename Reverse<List>::Type;

template <typename List, std::size_t N, typename NewItem>
struct Set;

template <typename Head, typename... Tail, std::size_t N, typename NewItem>
struct Set<TypeList<Head, Tail...>, N, NewItem> {
    using Type = Prepend_t<Head, typename Set<TypeList<Tail...>, N - 1, NewItem>::Type>;
};

template <typename Head, typename... Tail, typename NewItem>
struct Set<TypeList<Head, Tail...>, 0, NewItem> {
    using Type = TypeList<NewItem, Tail...>;
};

template <std::size_t N, typename NewItem>
struct Set<TypeList<>, N, NewItem> {
    // Always false if matched here
    static_assert(N < 0, "Index out of bounds in Set");
};

template <typename List, std::size_t N, typename NewItem>
using Set_t = typename Set<List, N, NewItem>::Type;

template <typename T, typename = void>
struct IsTensor : std::false_type {};

template <typename T>
struct IsTensor<T, std::void_t<typename T::IsTensor>> : std::true_type {};

template <typename T>
inline constexpr bool IsTensor_v = IsTensor<T>::value;

template <typename List, std::size_t pos, typename NewItem, typename = void>
struct UpdateOrPrepend {
    using Type = Prepend_t<NewItem, List>;
};

template <typename List, std::size_t pos, typename NewItem>
struct UpdateOrPrepend<List, pos, NewItem, std::enable_if_t<(pos < Size_v<List>)>> {
    using Type = Set_t<List, pos, NewItem>;
};

template <typename List, std::size_t pos, typename NewItem>
using UpdateOrPrepend_t = typename UpdateOrPrepend<List, pos, NewItem>::Type;

template <typename List>
struct Unique;

// 空列表特化
template <>
struct Unique<TypeList<>> {
    using Type = TypeList<>;
};

template <typename... Ts>
struct Unique<TypeList<Ts...>> {
private:
    // 辅助结构：在编译期构建去重列表
    template <typename Accumulator, typename... Remaining>
    struct TypeSet;

    // 基本情况：没有剩余元素
    template <typename... Accumulated>
    struct TypeSet<TypeList<Accumulated...>> {
        using Type = TypeList<Accumulated...>;
    };

    // 递归情况：处理下一个元素
    template <typename... Accumulated, typename Head, typename... Tail>
    struct TypeSet<TypeList<Accumulated...>, Head, Tail...> {
        // 检查Head是否已经在Accumulated中
        static constexpr bool contains = (std::is_same_v<Head, Accumulated> || ...);

        using Type = typename std::conditional_t<
            contains,
            // 如果包含，跳过Head
            TypeSet<TypeList<Accumulated...>, Tail...>,
            // 如果不包含，添加Head
            TypeSet<TypeList<Accumulated..., Head>, Tail...>>::Type;
    };

public:
    using Type = typename TypeSet<TypeList<>, Ts...>::Type;
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
struct Contains;

template <typename T>
struct Contains<TypeList<>, T> : std::false_type {};

template <typename... Tail, typename T>
struct Contains<TypeList<T, Tail...>, T> : std::true_type {};

template <typename Head, typename... Tail, typename T>
struct Contains<TypeList<Head, Tail...>, T> {
    static constexpr bool value = Contains<TypeList<Tail...>, T>::value;
};

template <typename List, typename T>
inline constexpr bool Contains_v = Contains<List, T>::value;

template <typename List1, typename List2>
struct Intersection {
    template <typename T>
    struct IsInList2 : Contains<List2, T> {};

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
constexpr auto ForEach(TypeList<>, Func&& /*func*/, Data data)
{
    return data;
}

template <typename Head, typename... Tail, typename Func, typename Data>
constexpr auto ForEach(TypeList<Head, Tail...>, Func&& func, Data data)
{
    return ForEach(TypeList<Tail...>{}, std::forward<Func>(func), std::forward<Func>(func)(TypeWrapper<Head>{}, data));
}

/*--------------------------------------------------------------------------------------------------------------------*/
/* A map with integer keys */
template <typename... Items>
class IndexMap {
private:
    using ItemList = TypeList<Items...>;

    template <size_t Index, typename Value>
    struct IndexValuePair {
        static constexpr size_t index = Index;
        using ValueType = Value;
    };

    /* Find */
    template <size_t I, typename List>
    struct FindByIndex;

    template <size_t I>
    struct FindByIndex<I, TypeList<>> {
        using Type = void;
        static constexpr bool found = false;
    };

    template <size_t I, typename Head, typename... Tail>
    struct FindByIndex<I, TypeList<Head, Tail...>> {
    private:
        using TailResult = FindByIndex<I, TypeList<Tail...>>;

    public:
        static constexpr bool found = (Head::index == I) || TailResult::found;

        using Type = std::conditional_t<(Head::index == I), typename Head::ValueType, typename TailResult::Type>;
    };

    /* Set */
    template <size_t I, typename NewType, typename List>
    struct SetByIndex;

    template <size_t I, typename NewType>
    struct SetByIndex<I, NewType, TypeList<>> {
        using Type = TypeList<IndexValuePair<I, NewType>>;
    };

    template <size_t I, typename NewType, typename Head, typename... Tail>
    struct SetByIndex<I, NewType, TypeList<Head, Tail...>> {
    private:
        using TailResult = typename SetByIndex<I, NewType, TypeList<Tail...>>::Type;

    public:
        using Type = std::conditional_t<
            (Head::index == I), Prepend_t<IndexValuePair<I, NewType>, TailResult>, Prepend_t<Head, TailResult>>;
    };

    template <typename List>
    struct ListToPack;

    template <typename... Ts>
    struct ListToPack<TypeList<Ts...>> {
        template <template <typename...> class T>
        using Apply = T<Ts...>;
    };

public:
    /* Find */
    template <size_t I>
    using At = typename FindByIndex<I, ItemList>::Type;

    /* Contains */
    template <size_t I>
    static constexpr bool Contains = FindByIndex<I, ItemList>::found;

    /* Set */
    template <size_t I, typename NewType>
    using Set = typename ListToPack<typename SetByIndex<I, NewType, ItemList>::Type>::template Apply<IndexMap>;

    /* Size */
    static constexpr size_t size = sizeof...(Items);
};
/*--------------------------------------------------------------------------------------------------------------------*/

} // namespace Atvoss::Util
#endif