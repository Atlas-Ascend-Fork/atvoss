# 表达式模板
## 1.概述
表达式模板是C++利用运算符重载，类型计算来实现延迟计算的技巧。其基本要点是：在形式上使用运算符或者进行函数的调用的时候，只是记录了要进行的操作，形成一颗类型化抽象语法树，并不会立即进行计算。在将来的某个时刻，可以对这个表达式进行求值计算，从而得到计算结果。
## 2.表达式模板的使用
```cpp
struct AddCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto in1 = Atvoss::PlaceHolder<1, Tensor<DtypeV1>, Atvoss::ParamUsage::in>();
        auto in2 = Atvoss::PlaceHolder<2, Tensor<DtypeV1>, Atvoss::ParamUsage::in>();
        auto out = Atvoss::PlaceHolder<3, Tensor<DtypeV2>, Atvoss::ParamUsage::out>();

        return (out = in1 + in2);
    }
};
```
以`AddCompute`为例，进行要素说明：  
- `AddCompute`类被设计为一个高度泛型的加法表达式，其输入类型`DtypeV1` 与`DtypeV2`由外部模板指定。  
- 为实现跨数据结构的通用性，其`Compute()`成员函数采用了模板模板参数，通过这一设计，只需传入不同的容器模板，即可派生出对于不同容器类型的加法计算，比如`vector<Dtype>`、`GlobalTensor<DType>` 或`LocalTensor<DType>`等容器类型的特化加法计算。该接口的作用是完成算子计算操作的描述，这里用户需要先定义输入输出，再定义基于输入输出的计算表达。  
- 用户定义的输入和输出需要使用`PlaceHolder()`进行创建，返回结果实质就是一个[Expression](#311-expression模板)节点，该节点内部封装了对应的[Param](#312-参数模板)参数。
- 定义好输入输出后，使用输入和输出描述计算过程，即上述例子中的`out = in1 + in2`。这里会得到一个封装着计算过程的`Expression<OpAssign<Param, OpAdd<...>>>`节点，后续延迟计算。运算具体实现见[OpAssign](#323-赋值运算符的使用)和[OpAdd](#322-基本运算符的使用)。


## 3.表达式模板原理介绍
### 3.1 基本表达式节点
#### 3.1.1 Expression模板
最基本的表达式节点是`Expression`模板，它的模板参数是任意类型：
```cpp
template <typename T>
struct Expression {
    static_assert(!std::is_rvalue_reference_v<T>, "[ERROR]: [Atvoss][Expression] Rvalue references cannot be stored");
    using Type = T;
    static constexpr bool hasData = HasDataTrait<T>::value;

    T const data{};

    template <typename U>
    [[nodiscard]] constexpr auto operator=(Expression<U> u);
};
```
- 作用：这是一个统一的包装器，任何元素（包括字面量、参数或运算操作）一经它封装，即成为表达式系统的一部分。具体来说，`Expression`的作用是将某种类型及其对应数据包装为一个表达式节点。`Expression`是构建计算图的基础单元，其本身仅负责信息存储，不执行实际计算。
- 约束：根据实际用法，类型`T`仅允许为值类型或左值引用类型，而不接受右值。
#### 3.1.2 参数模板
```cpp
namespace Atvoss {
template <std::size_t N, typename T, ParamUsage V = ParamUsage::in>
struct Param {
    using Type = T;
    static constexpr std::size_t number = N;
    static constexpr ParamUsage usage = V;
    static constexpr bool hasData = false;

    template <typename W>
    constexpr auto operator=(Expression<W>) {
        static_assert(Util::AlwaysFalse_v<W>, "[ERROR]: [Atvoss][Expression] Please use Expression<Param> for assignment");
    }
};
}
```
- 作用：用于表达在构建表达式时还没有实际值的参数。
- 约束：类型`T`仅允许为值类型或左值引用类型，而不接受右值。
- 模板参数介绍：  

    | 参数名       | 描述说明|
    | ------------ | ------------|
    | N | 输入输出的序号，对应`PlaceHolder`类中的序号，从1开始，依次加1递增，不能重复|
    | T | 数据类型， 若是容器类型，固定使用容器类型进行配置，若是Scalar类型，固定使用`T`进行配置，`T`为基础数据类型，如：float、half等|
    | V | 输入/输出类型标识符, 支持设置的类型为：`ParamUsage::in`、`ParamUsage::out`、`ParamUsage::in_out`|

#### 3.1.3 局部变量模板
```cpp
template <std::size_t N, typename T, typename L = void>
struct LocalVar {
    static_assert(!std::is_reference_v<T>,  "[ERROR]: [Atvoss][Expression] A LocalVar must not be a reference");
    using Type = T;
    using Like = L;
    static constexpr std::size_t number = N;
    static constexpr bool hasData = false;

    template <typename V>
    constexpr auto operator=(Expression<V>) {
        static_assert(Util::AlwaysFalse_v<V>, "[ERROR]: [Atvoss][Expression] Please use Expression<LocalVar> for assignment");
    }
};
```
- 作用：用于提供表达式内部的临时变量支持，实现了类似函数局部变量的功能。由于局部变量需要预分配存储空间，可通过`L/Like`标识声明其与某个参数具有相似性，以便在初始化时参照该参数的规格（例如`size()`）进行初始化。
- 约束：目前不支持引用类型的局部变量。
- 模板参数介绍： 
  

    | 参数名| 描述说明|
    | ------------ | ------------|
    | N | 临时Buffer的序号, 序号从1开始，依次加1递增，不能重复|
    | L | 入参的`Expression`包含的模板参数，固定由入参类型推倒，用户无需关注|
### 3.2 表达式模板的组合
#### 3.2.1 运算符基类
二元运算符基类如下：
```cpp
template <typename T, typename U>
struct BinaryOp : private Util::CompressedPair<T, U> {
private:
    using Storage = Util::CompressedPair<T, U>;

public:
    static_assert(!(std::is_rvalue_reference_v<T> ||
                    std::is_rvalue_reference_v<U>),
                  "[ERROR]: [Atvoss][Expression] Rvalue references cannot be stored");
    static constexpr bool hasData =
            HasDataTrait<T>::value || HasDataTrait<U>::value;
    using IsBinaryOp = void;
    using LhsType = T;
    using RhsType = U;

    BinaryOp() = default;
    constexpr BinaryOp(T t, U u) : Storage(t, u) {}

    constexpr const T& GetLhs() const { return Storage::First(); }
    constexpr const U& GetRhs() const { return Storage::Second(); }
};
```
考虑到实际计算的时候`T`和`U`往往是空类，该类通过继承`CompressedPair`进而实现空基类优化，具体实现参考[common.h](../include/utils/expression/common.h)。接下来针对该类的模板参数，成员变量和成员函数进行介绍：

- 模板参数介绍：

    | 参数名| 描述说明| 
    | ------------ | ------------|
    | T | 左操作数的类型，即`LhsType`|
    | U | 右操作数的类型，即`RhsType`|


- 成员变量介绍：

    | 变量名| 描述说明|
    | ------------ | ------------|
    | hasData | 判断当前节点是否存储实际数据，用于后续编译期计算|
    | IsBinaryOp | 判断当前操作是否是二元操作，用于后续编译期计算|
- 成员函数介绍：

    | 函数名| 描述说明|
    | ------------ | ------------|
    | GetLhs | 获取左操作数|
    | GetRhs | 获取右操作数|

一元运算符基类设计如下，其和二元运算基类类似，故这里不做重复说明，具体实现参考[common.h](../include/utils/expression/common.h)：
```cpp
template <typename T>
struct UnaryOp : private Util::CompressedData<T> {
private:
    using Storage = Util::CompressedData<T>;

public:
    static_assert(!std::is_rvalue_reference_v<T>,
                  "[ERROR]: [Atvoss][Expression] Rvalue references cannot be stored");
    static constexpr bool hasData = HasDataTrait<T>::value;
    using IsUnaryOp = void;
    using DataType = T;

    UnaryOp() = default;
    constexpr UnaryOp(T t) : Storage(t) {}

    constexpr const T& GetData() const { return Storage::Data(); }
};
```
#### 3.2.2 基本运算符的使用
最基本的表达式模板组合就是使用加减乘除运算符，以加法为例，具体实现参考[math.h](../include/utils/expression/math.h)：
```cpp
template<typename T, typename U>
__host_aicore__ constexpr auto operator+(Expression<T> lhs, Expression<U> rhs) 
{
    return Expression<OpAdd<T, U>>{{lhs.data, rhs.data}};
}
```
使用上述的运算符重载，当编译器遇到`expr1 + expr2`时，不会直接执行加法运算，而是返回一个表示加法操作的`Expression`对象，其内部类型通常为 `OpAdd`（类似的也可以是`OpSub`、`OpMul`、`OpDiv`等）。此处以`OpAdd`为例说明其结构。

`OpAdd`包含两个模板参数，并拥有两个对应的数据成员。借助`BinaryOp`，该结构可简化为如下形式：
```cpp
template<typename T, typename U>
struct OpAdd : BinaryOp<T, U> {
    OpAdd() = default;
    constexpr OpAdd(T t, U u) : BinaryOp<T, U>(t, u) {}
};
```

#### 3.2.3 赋值运算符的使用
为了表达向出参和局部变量赋值，支持赋值运算符，基本的运算符重载如下，具体实现参考[common.h](../include/utils/expression/common.h)：
```cpp
template <typename T>
template <typename U>
__host_aicore__ constexpr auto Expression<T>::operator=(Expression<U> u)
{
    static_assert(
            (IsParam_v<T> || IsLocalVar_v<T> || std::is_lvalue_reference_v<T>),
            "[ERROR]: [Atvoss][Expression] Only a Param, LocalVar, or reference can appear on the left side of assignment");
    return Expression<OpAssign<T, U>>{{data, u.data}};
}
```
对应的操作类型为`OpAssign`，结构如下：
```cpp
struct OpAssign : BinaryOp<T, U> {
    OpAssign() = default;
    constexpr OpAssign(T t, U u) : BinaryOp<T, U>(t, u) {}
};
```


#### 3.2.4 表达序列
为了表达过程式的操作，支持使用`,`运算符表达操作序列。基本的运算符重载如下：
```cpp
template <typename T, typename U>
__host_aicore__ constexpr auto operator,(Expression<T> t, Expression<U> u)
{
    return Expression<OpAndThen<T, U>>{{t.data, u.data}};
}
```
上述重载允许表达式的形式为`tmp1 = in1 + in2, out = tmp1 + tmp1`。对应的表达式的操作类型为`OpAndThen`，结构如下：
```cpp
template <typename T, typename U>
struct OpAndThen : BinaryOp<T, U> {
    OpAndThen() = default;
    constexpr OpAndThen(T t, U u) : BinaryOp<T, U>(t, u) {}
};
```
### 3.3 编译期参数收集
通过类型列表操作，系统在编译的时候，分析表达式中使用的所有参数（`Param`
）。
```cpp
template <typename T>
struct Params {
    using Type = typename Detail::UniqueParams<T>::Type;
}
```
该设计允许系统：
+ 自动推导需要的参数类型和数量。
+ 为参数分配存储空间。
+ 在编译时校验变量使用的正确性。
### 3.4 计算
截止当前，已经可以使用表达式模板组合出计算过程，但是还没有实际的代码支持计算操作。本模板系统计算的核心是通过`Evaluator`实现的，具体结构如下：
```cpp
template <typename T>
struct Evaluator {
    using Type = T;

    template <typename ArgTup, typename LocalVarTup>
    __aicore__ decltype(auto) operator()(const T& value, ArgTup& args, LocalVarTup& localVars) const
    {
        ...
    }
};

```
使用`operator()`来进行求值，使用`Type`表示求值的结果类型。  
求值过程采用递归的深度优先遍历策略：  
从根节点开始，逐一对每棵子树进行求值。针对每种支持的操作类型，需在实现中提供相应的特化版本，通过递归方式完成求值。以`Evaluator<OpAdd<T, U>>`为例，其求值逻辑如下，具体实现参考[tile_evaluator_math.h](../include/tile/tile_evaluator_math.h)：
1. 创建`Evaluator<T>`对左子节点`op.lhs`进行求值；
2. 创建`Evaluator<U>`对右子节点`op.rhs`进行求值；
3. 将两个子节点的结果相加。

终止条件：递归过程在引用/常量、`Evaluator<Param<…>>`或 `Evaluator<LocalVar<…>>`处终止，不再继续向下递归，而是直接从对应的引用/常量、运行时提供的参数元组`args`或局部变量元组`localVars`中获取相应值。
若缺少对应的特化实现，则求值过程会在编译阶段报错。


### 3.5 辅助工具：编译期类型操作
代码中包含了一系列的模板元编程（TMP）工具，是构建整个系统的“标准库”，用于在编译期对类型列表进行计算。
现对这些主要用到的工具进行简单介绍，具体实现参考[utility.h](../include/utils/expression/utility.h)：
  
  `TypeList`：一个用于容纳零个或多个类型的“容器”，类似于`std::tuple`但只在类型层面工作。

针对`TypeList`的常规操作（很多源自函数式编程），有如下操作：
+ `Append_t`：在`TypeList`的结尾增加一个类型。
+ `Concatenate_t`：将多个`TypeList`连接成一个。
+ `Filter_t`：根据一个谓词（Predicate）模板，从`TypeList`中过滤出符合条件的类型。
+ `Find_v`：在`TypeList`中查找第一个满足谓词的类型。如果不存在，则得到`TypeList`的长度。
+ `FindLast_v`：在`TypeList`中查找最后一个满足谓词的类型。如果不存在，则得到`TypeList`的长度。
+ `FindUnique_t`：在`TypeList`中查找唯一一个满足谓词的类型。如果不存在，或存在多个，都会造成编译失败。
+ `Get_t`：得到`TypeList`中指定位置的类型。
+ `Map_t`：对`TypeList`中的每一项类型都执行一个映射动作，结果仍然是个`TypeList`。
+ `Prepend_t`：在`TypeList`的开头增加一个类型。
+ `Set_t`：更新`TypeList`中指定位置的类型。
+ `Unique_t`：移除`TypeList`中的重复类型。
+ `Contains`：查找`TypeList`是否包含指定的类型。
+ `Intersection`：求出两个`TypeList`的交集。
+ `Difference`：求出两个`TypeList`的差集，即只存在于第一个`TypeList`，而不存在于第二个`TypeList`中的类型。
+ `ForEach`：给定一个`TypeList`、一个函数对象`func`和一个初始对象，以`TypeList`中的每一项作为`func`的第一个参数，并以初始对象或上次求值的结果作为第二个参数，持续使用`func`求值，并返回最后的结果。
