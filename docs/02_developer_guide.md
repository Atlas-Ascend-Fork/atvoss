# 1. ATVOSS概述
ATVOSS针对昇腾AI处理器的Vector计算的多核并行计算过程，提供了一套统一的编程模型，可简化用户开发Vector算子的复杂度，简化Tiling计算，并支持模板的自由组合及模板扩展。

编程模型包含如下分层，由高到低分别是：

- Device层：Host侧调用总入口，完成参数校验、Acl资源管理、Host侧与Device侧的数据管理、切分计算、Workspace管理、Kernel调用等逻辑。
- Kernel层：Kernel函数总入口，负责多核间的任务分解，控制Block的调度。
- Block层：负责单核的任务分解到多个Tile块，控制数据搬运/计算数据流进行流水编排处理。
- Tile层：对Ascend C基础API进行封装，提供更大Tile块的搬运、计算等能力。
- Basic层：使用Ascend C基础API能力完成数据搬运计算等基础操作。
<br><img src="./images/architecture.png" width="50%" height="50%" style="margin: 20px 0;"><br>
# 2. 公共基础工具
## 2.1 表达式模板
ATVOSS使用表达式模板进行计算流的描述，表达式模板是C++利用运算符重载、类型计算来实现延迟计算的技巧。表达式模板的基本要点是在形成使用运算符或进行函数调用时，在Kernel/Block/Tile层使用表达式模板，只是萃取表达式包含的计算操作信息，直到最后Tile层计算的时候，才会进行实际的计算操作。详细的表达式模板实现，可从[Expression表达式模板](../include/utils/expression/expression.h)进入阅读代码。

# 3. 基于ATVOSS的算子开发
## 3.1 实现自定义计算表达式
ATVOSS模板支持Expr的计算表达，用户通过计算表达完成对自定义算子的描述。具体使用样例如下：
```cpp
using T = float;

struct RmsNormCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto in1 = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        auto in2 = Atvoss::PlaceHolder<2, Tensor<T>, Atvoss::ParamUsage::in>();
        auto out = Atvoss::PlaceHolder<3, Tensor<T>, Atvoss::ParamUsage::out>();
        auto temp = Atvoss::PlaceHolderTmpLike<1>(in1);

        return (temp = in1 * in1,
                out = ReduceSum<Atvoss::Pattern::AR>(temp),
                out = Broadcast<Atvoss::Pattern::AB>(out),
                temp = Divs<WIDTH>(out),
                out = Sqrt(temp),
                temp = in1 / out,
                out = in2 * temp);
    }
};
```

以上代码为例，定义一个`RmsNormCompute`的Expr模板类, 模板类中固定提供一个`Compute()`接口，接口定义固定格式如下：
```cpp
template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        // 定义输入输出等参数描述
        auto in1 = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        ...
        // 返回计算表达式
        return (...);
    }
```
`Compute()`接口完成算子计算操作的描述，需要先定义输入输出，再定义基于输入输出的计算表达。

### 3.1.1 定义输入输出
当前支持的表达式参数定义如下：

#### 3.1.1.1 PlaceHolder
- 功能说明

    描述输入输出信息。

- 参数定义原型

    ```cpp
    template <std::size_t N, typename T,  ParamUsage V = ParamUsage::in>
        __host_aicore__ constexpr auto PlaceHolder()
    ```
- 模板参数

    | 参数名| 描述说明|
    | ------------ | ------------|
    | N | 输入输出的序号，`PlaceHolder`类的序号从1开始，依次加1递增，不能重复|
    | T | 数据类型， `PlaceHolder`若是Tensor类型， 固定使用`Tensor<T>`进行配置，若是Scalar类型，固定使用`T`进行配置，`T`为基础数据类型，如：float， half等。|
    | V | 输入/输出类型标识符, 支持设置的类型为：`ParamUsage::in`、`ParamUsage::out`、`ParamUsage::in_out`|

- 返回值说明

    返回一个`Expression<ExprTmpl::Param<N ,T, V>>`型的表达式模板

#### 3.1.1.2 PlaceHolderTmpLike

- 功能说明

    描述临时Buffer的信息。

- 参数定义原型

    ```cpp
    template <std::size_t N, typename L>
        __host_aicore__ constexpr auto PlaceHolderTmpLike(ExprTmpl::Expression<L>) {
        }
    ```

- 模板参数

    | 参数名| 描述说明|
    | ------------ | ------------|
    | N | 临时Buffer的序号, `PlaceHolderTmpLike`类的序号从1开始，依次加1递增，不能重复|
    | L | 入参的`Expression`包含的模板参数，固定由入参类型推倒，用户无需关注|

- 参数说明

    | 参数名| 描述说明|
    | ------------ | ------------|
    | `Expression<L>` | 固定配置为某个`PlaceHolder`, 表征需要给本临时Buffer申请的内存的大小与该`PlaceHolder`申请的内存大小一样|


- 返回值说明

    返回一个`ExprTmpl::Expression<ExprTmpl::LocalVar<N, typename L::Type, L>>{}`型的表达式模板

### 3.1.2 定义计算表达式

使用Tile层API接口名描述计算过程, 每次使用1个Tile层的API操作实现一次计算过程，每个计算过程是按“,”隔开，用户需要保证计算表达的正确性。


## 3.2 Tile层API
1个Tile层API有两个接口呈现方式，两者相互对应， 用户在计算图中使用API接口名来描述计算操作过程，Tile层内部则是调用实现接口完成实际的计算过程。

- API接口名

    是用户在计算图中调用时使用的接口名，例如`Sqrt()`接口。
    ```cpp
    _2 = Sqrt(_1) // _1 是Sqrt()的输入，_2是输出
    ```

- 实现接口名

    是Tile层的真正执行计算时调用的接口定义。

- 已支持的Tile层API列表

    | API接口名    | 实现接口名                                    | 备注                            |
    |:----------|:-----------------------------------------|:------------------------------|
    | +         | [AddAssign](#321-AddAssign)              | 按元素求和                         |
    | -         | [SubAssign](#322-SubAssign)              | 按元素求差                         |
    | *         | [MulAssign](#323-MulAssign)              | 按元素求积                         |
    | /         | [DivAssign](#324-DivAssign)              | 按元素求商                         |
    | Divs      | [DivsAssign](#325-DivsAssign)            | 矢量内每个元素与标量求积                  |
    | Exp       | [ExpAssign](#326-ExpAssign)              | 按元素取自然指数                      |
    | Sqrt      | [SqrtAssign](#327-SqrtAssign)            | 按元素做开方                        |
    | Power     | [PowerAssign](#328-PowerAssign)          | 实现按元素做幂运算功能，目前只支持平方           |
    | Broadcast | [BroadcastAssign](#329-BroadcastAssign)  | 将输入按照输出shape进行广播，目前只支持二维      |
    | ReduceSum | [ReduceSumAssign](#3210-ReduceSumAssign) | 对一个多维向量按照指定的维度进行数据累加，目前只支持二维  |
    | Cast      | [CastAssign](#3211-CastAssign)           | 根据源操作数和目的操作数Tensor的数据类型进行精度转换 |

### 3.2.1 AddAssign
#### 3.2.1.1 功能说明
按元素求和，计算公式如下：
$$
dst_i=src0_i + src1_i
$$
#### 3.2.1.2 函数原型
```cpp
template <typename OperationShape, typename T>
__aicore__ inline void AddAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0,
                                 const AscendC::LocalTensor<T>& src1, OperationShape& operationShape)
```
#### 3.2.1.3 模版参数说明
| 参数名            | 描述                          |
|----------------|-----------------------------|
| OperationShape | 数据长度结构体的类型                  |
| T              | 数据类型，支持的数据类型为：int32_t/float |

#### 3.2.1.4 参数说明
| 参数名            | 输入/输出 | 描述           |
|----------------|-------|--------------|
| dst            | 输出    | 目的操作数        |
| src0、src1      | 输入    | 源操作数         |
| operationShape | 输入    | 需要计算数据长度的结构体 |
#### 3.2.1.5 表达式调用示例
```c++
using TileShape = Atvoss::Shape<WIDTH>;

template<typename T>
struct AddCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto src0 = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        auto src1 = Atvoss::PlaceHolder<2, Tensor<T>, Atvoss::ParamUsage::in>();
        auto dst = Atvoss::PlaceHolder<3, Tensor<T>, Atvoss::ParamUsage::out>();
        return (dst = src0 + src1);
    }
};
```
`operationShape`用户无需感知。

### 3.2.2 SubAssign
#### 3.2.2.1 功能说明
按元素求差，计算公式如下：
$$
dst_i=src0_i - src1_i
$$
#### 3.2.2.2 函数原型
```cpp
template <typename OperationShape, typename T>
__aicore__ inline void SubAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0,
                                 const AscendC::LocalTensor<T>& src1, OperationShape& operationShape)
```
#### 3.2.2.3 模版参数说明
| 参数名            | 描述                          |
|----------------|-----------------------------|
| OperationShape | 数据长度结构体的类型                  |
| T              | 数据类型，支持的数据类型为：int32_t/float |

#### 3.2.2.4 参数说明
| 参数名            | 输入/输出 | 描述           |
|----------------|-------|--------------|
| dst            | 输出    | 目的操作数        |
| src0、src1      | 输入    | 源操作数         |
| operationShape | 输入    | 需要计算数据长度的结构体 |
#### 3.2.2.5 表达式调用示例
```c++
using TileShape = Atvoss::Shape<WIDTH>;

template<typename T>
struct SubCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto src0 = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        auto src1 = Atvoss::PlaceHolder<2, Tensor<T>, Atvoss::ParamUsage::in>();
        auto dst = Atvoss::PlaceHolder<3, Tensor<T>, Atvoss::ParamUsage::out>();
        return (dst = src0 - src1);
    }
};
```
`operationShape`用户无需感知。

### 3.2.3 MulAssign
#### 3.2.3.1 功能说明
按元素求积，计算公式如下：
$$
dst_i=src0_i * src1_i
$$
#### 3.2.3.2 函数原型
```cpp
template <typename OperationShape, typename T>
__aicore__ inline void MulAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0,
                                 const AscendC::LocalTensor<T>& src1, OperationShape& operationShape)
```
#### 3.2.3.3 模版参数说明
| 参数名            | 描述                          |
|----------------|-----------------------------|
| OperationShape | 数据长度结构体的类型                  |
| T              | 数据类型，支持的数据类型为：int32_t/float |

#### 3.2.3.4 参数说明
| 参数名            | 输入/输出 | 描述           |
|----------------|-------|--------------|
| dst            | 输出    | 目的操作数        |
| src0、src1      | 输入    | 源操作数         |
| operationShape | 输入    | 需要计算数据长度的结构体 |
#### 3.2.3.5 表达式调用示例
```c++
using TileShape = Atvoss::Shape<WIDTH>;

template<typename T>
struct MulCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto src0 = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        auto src1 = Atvoss::PlaceHolder<2, Tensor<T>, Atvoss::ParamUsage::in>();
        auto dst = Atvoss::PlaceHolder<3, Tensor<T>, Atvoss::ParamUsage::out>();
        return (dst = src0 * src1);
    }
};
```
`operationShape`用户无需感知。

### 3.2.4 DivAssign
#### 3.2.4.1 功能说明
按元素求商，计算公式如下：
$$
dst_i=src0_i/src1_i
$$
#### 3.2.4.2 函数原型
```cpp
template <typename OperationShape, typename T>
__aicore__ inline void DivAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0,
                                 const AscendC::LocalTensor<T>& src1, OperationShape& operationShape)
```
#### 3.2.4.3 模版参数说明
| 参数名            | 描述                  |
|----------------|---------------------|
| OperationShape | 数据长度结构体的类型          |
| T              | 数据类型，支持的数据类型为：float |

#### 3.2.4.4 参数说明
| 参数名            | 输入/输出 | 描述           |
|----------------|-------|--------------|
| dst            | 输出    | 目的操作数        |
| src0、src1      | 输入    | 源操作数         |
| operationShape | 输入    | 需要计算数据长度的结构体 |
#### 3.2.4.5 表达式调用示例
```c++
using TileShape = Atvoss::Shape<WIDTH>;

template<typename T>
struct DivCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto src0 = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        auto src1 = Atvoss::PlaceHolder<2, Tensor<T>, Atvoss::ParamUsage::in>();
        auto dst = Atvoss::PlaceHolder<3, Tensor<T>, Atvoss::ParamUsage::out>();
        return (dst = src0 / src1);
    }
};
```
`operationShape`用户无需感知。

### 3.2.5 DivsAssign
#### 3.2.5.1 功能说明
矢量内每个元素与标量求商，计算公式如下：
$$
dst_i=src_i/scalar
$$
#### 3.2.5.2 函数原型
```cpp
template <typename OperationShape, auto scalarValue, typename T>
__aicore__ inline void DivsAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
                                  OperationShape& operationShape)
```
#### 3.2.5.3 模版参数说明
| 参数名            | 描述                          |
|----------------|-----------------------------|
| OperationShape | 数据长度结构体的类型                  |
| scalarValue    | 源操作数，作为除数                   |
| T              | 数据类型，支持的数据类型为：int32_t/float |

#### 3.2.5.4 参数说明
| 参数名            | 输入/输出 | 描述           |
|----------------|-------|--------------|
| dst            | 输出    | 目的操作数        |
| src            | 输入    | 源操作数         |
| operationShape | 输入    | 需要计算数据长度的结构体 |
#### 3.2.5.5 表达式调用示例
```c++
using TileShape = Atvoss::Shape<WIDTH>;

template<typename T>
struct DivsCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto src = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        auto dst = Atvoss::PlaceHolder<2, Tensor<T>, Atvoss::ParamUsage::out>();
        return (dst = Divs<2>(src));
    } 
};
```
`operationShape`用户无需感知。

### 3.2.6 ExpAssign
#### 3.2.6.1 功能说明
按元素取自然指数，计算公式如下：
$$
dst_i=e^{src_i}
$$
#### 3.2.6.2 函数原型
```cpp
template <typename OperationShape, typename T>
__aicore__ inline void ExpAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
                                 OperationShape& operationShape)
```
#### 3.2.6.3 模版参数说明
| 参数名       | 描述                  |
|-----------|---------------------|
| T         | 数据类型，支持的数据类型为：float |

#### 3.2.6.4 参数说明
| 参数名            | 输入/输出 | 描述           |
|----------------|-------|--------------|
| dst            | 输出    | 目的操作数        |
| src            | 输入    | 源操作数         |
| operationShape | 输入    | 需要计算数据长度的结构体 |
#### 3.2.6.5 表达式调用示例
```c++
using TileShape = Atvoss::Shape<WIDTH>;

template<typename T>
struct ExpCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto src = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        auto dst = Atvoss::PlaceHolder<2, Tensor<T>, Atvoss::ParamUsage::out>();
        return (dst = Exp(src));
    }
};
```
`operationShape`用户无需感知。

### 3.2.7 SqrtAssign
#### 3.2.7.1 功能说明
按元素做开方，计算公式如下：
$$
dst_i=\sqrt{src_i}
$$
#### 3.2.7.2 函数原型
```cpp
template <typename OperationShape, typename T>
__aicore__ inline void SqrtAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
                                  OperationShape& operationShape)
```
#### 3.2.7.3 模版参数说明
| 参数名            | 描述                  |
|----------------|---------------------|
| OperationShape | 数据长度结构体的类型          |
| T              | 数据类型，支持的数据类型为：float |

#### 3.2.7.4 参数说明
| 参数名            | 输入/输出 | 描述           |
|----------------|-------|--------------|
| dst            | 输出    | 目的操作数        |
| src            | 输入    | 源操作数         |
| operationShape | 输入    | 需要计算数据长度的结构体 |
#### 3.2.7.5 表达式调用示例
```c++
using TileShape = Atvoss::Shape<WIDTH>;

template<typename T>
struct SqrtCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto src = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        auto dst = Atvoss::PlaceHolder<2, Tensor<T>, Atvoss::ParamUsage::out>();
        return (dst = Sqrt(src));
    }
};
```
`operationShape`用户无需感知。

### 3.2.8 PowerAssign
#### 3.2.8.1 功能说明
按元素做幂运算（目前只支持平方），计算公式如下：
$$
dst_i=src_i^{scalarValue}
$$
#### 3.2.8.2 函数原型
```cpp
template <typename OperationShape, auto scalarValue, typename T>
__aicore__ inline void PowerAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
                                   OperationShape& operationShape)
```
#### 3.2.8.3 模版参数说明
| 参数名            | 描述                  |
|----------------|---------------------|
| OperationShape | 数据长度结构体的类型          |
| scalarValue    | 源操作数，作为指数，只支持平方     |
| T              | 数据类型，支持的数据类型为：float |

#### 3.2.8.4 参数说明
| 参数名            | 输入/输出 | 描述           |
|----------------|-------|--------------|
| dst            | 输出    | 目的操作数        |
| src            | 输入    | 源操作数         |
| operationShape | 输入    | 需要计算数据长度的结构体 |
#### 3.2.8.5 表达式调用示例
```c++
using TileShape = Atvoss::Shape<WIDTH>;

template<typename T>
struct PowerCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto src = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        auto dst = Atvoss::PlaceHolder<2, Tensor<T>, Atvoss::ParamUsage::out>();
        return dst = (Power<2>(src));
    }
};
```
`operationShape`用户无需感知。

### 3.2.9 BroadcastAssign
#### 3.2.9.1 功能说明
将输入按照输出Shape进行广播，（目前只支持二维）。
比如A的Shape为(2, 1)，广播的目标Shape为(2, 16)，则会将原来的一列扩展为相同的16列。
```c++
输入数据： 
[[ 1]
 [ 2]]
输出数据： 
[[ 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1]
 [ 2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2]]
```
#### 3.2.9.2 函数原型
```cpp
template <typename OperationShape, Atvoss::Pattern pattern, typename T>
__aicore__ inline void BroadcastAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
                                       OperationShape& operationShape)
```
#### 3.2.9.3 模版参数说明
| 参数名            | 描述                      |
|----------------|-------------------------|
| OperationShape | 需要广播数据的维度类型             |
| pattern        | AB表示第1维进行广播，BA表示第0维进行广播 |
| T              | 数据类型，支持的数据类型为：float     |

#### 3.2.9.4 参数说明
| 参数名            | 输入/输出 | 描述           |
|----------------|-------|--------------|
| dst            | 输出    | 目的操作数        |
| src            | 输入    | 源操作数         |
| operationShape | 输入    | 需要计算数据长度的结构体 |
#### 3.2.9.5 表达式调用示例
```c++
using TileShape = Atvoss::Shape<WIDTH>;

template<typename T>
struct BroadcastCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto src = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        auto dst = Atvoss::PlaceHolder<2, Tensor<T>, Atvoss::ParamUsage::out>();
        return (dst = Broadcast<Atvoss::Pattern::AB>(src));
    }
};
```
`operationShape`用户无需感知。

### 3.2.10 ReduceSumAssign
#### 3.2.10.1 功能说明
对一个多维向量按照指定的维度进行数据累加（目前只支持二维）。
定义指定计算的维度（Reduce轴）为R轴，非指定维度（Normal轴）为A轴。如下图所示，对Shape为(2, 3)的二维矩阵进行运算，指定在第一维计算数据的累加，输出结果为[[5, 7, 9]]；指定在第二维计算数据的累加，输出结果为[[6] [15]]。
ReduceSum按最后一个维度计算示例：

<table>
<tr><td colspan="3"> 原始数据</td>  <td  rowspan="3" > -> </td><td>结果</td></tr>
<tr><td>1</td><td>2</td><td>3</td> <td>6</td></tr>
<tr><td>4</td><td>5</td><td>6</td> <td>15</td></tr>
</table>

#### 3.2.10.2 函数原型
```cpp
template <typename OperationShape, Atvoss::Pattern pattern, typename T>
__aicore__ inline void ReduceSumAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
                                       OperationShape& operationShape)
```
#### 3.2.10.3 模版参数说明
| 参数名            | 描述                    |
|----------------|-----------------------|
| OperationShape | reduce数据的维度类型         |
| pattern        | AR在第1维进行累加，RA在第0维进行累加 |
| T              | 数据类型，支持的数据类型为：float   |

#### 3.2.10.4 参数说明
| 参数名            | 输入/输出 | 描述            |
|----------------|-------|---------------|
| dst            | 输出    | 目的操作数         |
| src            | 输入    | 源操作数          |
| operationShape | 输入    | reduce数据的维度数据 |
#### 3.2.10.5 表达式调用示例
```c++
using TileShape = Atvoss::Shape<WIDTH>;

template<typename T>
struct ReduceSumCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto src = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        auto dst = Atvoss::PlaceHolder<2, Tensor<T>, Atvoss::ParamUsage::out>();
        return (dst = ReduceSum<Atvoss::Pattern::AR>(src));
    }
};
```
`operationShape`用户无需感知。

### 3.2.11 CastAssign
#### 3.2.11.1 功能说明
根据源操作数和目的操作数Tensor的数据类型进行精度转换

#### 3.2.11.2 函数原型
```cpp
template <typename OperationShape, CastMode castMode, typename T1, typename T2>
__aicore__ inline void CastAssign(AscendC::LocalTensor<T1>& dst, const AscendC::LocalTensor<T2>& src,
                                  OperationShape& operationShape)
```
#### 3.2.11.3 模版参数说明
| 参数名            | 描述                                    |
|----------------|---------------------------------------|
| OperationShape | 数据长度结构体的类型                            |
| castMode       | 精度转换处理模式                              |
| T1             | 目的操作数数据类型，支持的数据类型为：half、float、int32_t |
| T2             | 源操作数数据类型，支持的数据类型为：half、float、int32_t  |

#### 3.2.11.4 参数说明
| 参数名            | 输入/输出 | 描述           |
|----------------|-------|--------------|
| dst            | 输出    | 目的操作数        |
| src            | 输入    | 源操作数         |
| operationShape | 输入    | 需要计算数据长度的结构体 |
#### 3.2.11.5 表达式调用示例
```c++
using TileShape = Atvoss::Shape<WIDTH>;

template<typename T, typename U>
struct CastCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto src = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::in>();
        auto dst = Atvoss::PlaceHolder<2, Tensor<U>, Atvoss::ParamUsage::out>();
        return (dst = Cast<Atvoss::CastMode::CAST_ROUND>(src));
    }
};
```
`operationShape`用户无需感知。

## 3.3 配置Block层模板
Block层模板负责单个核的任务分解到多个Tile的数据搬运、Tile层的计算调用。

### 3.3.1 BlockBuilder模板
```cpp
template <typename Compute, const auto& Policy = blockPolicyDefault, typename ScheduleCfg = BlockConfig,
          class Schedule = Block::DefaultSchedule<Compute, Policy, ScheduleCfg>>
class BlockBuilder {
public:
    // 调度策略模板类
    using ScheduleClz = Schedule;
    // Block层执行函数，调用对应调度策略的执行函数
    template <typename ArgTup>
    __aicore__ inline void Run(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        Schedule schedule;
        schedule.Run(cfg, argTuple);
    }
};
```
#### 3.3.1.1 BlockBuilder模板的范式表达
`BlockBuilder`模板对外提供三个部分内容：
- 模板参数  

| 参数名                             | 描述                |
|---------------------------------|-------------------|
| [Compute](#31-实现自定义计算表达式)     | 自定义计算表达式          |
| [Policy](#33121-policy)         | Block层UB相关配置项     |
| [ScheduleCfg](#33122-schedulecfg)   | Block层数据元素切分结果配置项 |
| [Schedule](#33123-schedule) | Block层切分算法        |
- 默认构造函数
```cpp
// 默认构造函数：根据表达式最大存活节点中输入、输出元素个数进行UB内存分配的管理，主要实现在utils/buf_pool/loopbuf.h中
_aicore__ inline BlockBuilder();
```
- 对外运行接口`Run`
```cpp
// Block层执行函数。
// 调用Schedule的执行函数
template<typename ArgTup>
__aicore__ inline void Run(Arguments &cfg, ArgTup &argTuple);
```
#### 3.3.1.2 相关参数说明
#### 3.3.1.2.1 Policy
`Policy`：配置的是针对UB的使用，以及N对齐的数据处理策略。
```cpp
template <typename Shape>
struct BlockPolicy {
    uint32_t ubSizeMax = 190 * 1024;  // 最大使用的UB空间大小，默认190K
    Shape tileShape{};                // 一维: 不需要额外对齐，二维：按Shape的第二个配置值对齐
};
```
#### 3.3.1.2.2 ScheduleCfg
`ScheduleCfg`：根据不同策略计算出来的`Tile`层处理的元素信息。
```cpp
// Block层的元素切分信息
struct BlockConfig {
    uint32_t wholeLoop = 0;    // 当前Block中切分的Tile整块个数（不包含尾块）
    uint32_t tileCnt = 0;      // 当前Block是尾块时，处理的输入元素数量。整块时为零
    uint32_t basicNum = 0;     // 当前Block中分配的Tile块处理的输入元素数量
    uint32_t totalElemCnt = 0; // 当前Block块处理的输入元素总数量数
    UbAssign ubAssign;         // UB 空间分配策略
};

// 输入输出UB空间分配策略
struct UbAssign {
    uint32_t ubInNum = 1;               // 输入需要占用的UB tensor 个数
    uint32_t ubOutCnt = 1;              // 输出需要占用的UB tensor 个数
    uint32_t ubTmpCnt = 0;              // tmp需要占用的UB tensor 个数
    uint32_t eleNumSingleTensor = 1024; // 一个in tensor的元素个数
};
```
#### 3.3.1.2.3 Schedule
`Schedule`是计算任务的调度核心，它调用对应的切分策略来执行计算。在扩展性方面，用户可以通过继承`BaseBlockSchedule`基类可实现自定义策略；系统默认的执行策略为`DefaultSchedule`。
- 编译期根据[Policy](#33121-policy)参数配置，计算单个存活节点占用`UB`空间。
```cpp
// 编译期获取单个存活节点占用UB空间大小
static constexpr uint32_t GetEleCntInTensor();
```
- 编译期计算Block层配置，作为默认[ScheduleCfg](#33122-schedulecfg)参数
```cpp
// Schedule层默认配置计算
static constexpr bool MakeBlockParam(Arguments &blockParam);
```

- 运行接口`Run`
```cpp
// Block层调用的实际执行函数。
// cfg切分结果配置，通过单Block最大处理元素个数（totalElemCnt）计算Tile整块与尾块个数
// argTuple为GM侧的输入、输出数据空间
template<typename ArgTup>
__aicore__ inline void Run(Arguments &cfg, ArgTup &argTuple);
```

#### 3.3.1.3 模板对外接口说明
计算逻辑的实际执行是通过`BaseBlockSchedule`类中的Run方法实现。接下来对用户可感知的`BaseBlockSchedule`类中的核心函数进行介绍。
#### 3.3.1.3.1 Run接口
`BaseBlockSchedule`提供`Run()`接口按批次完成输入数据从GM->UB，计算，输出结果UB->GM过程，参数和函数说明如下：
- 参数说明：接收切分配置`cfg`，以及`Kernel`层传入的输入、输出GM的地址`argTuple`。
- 函数功能说明：通过Block需要处理的总元素个数（`cfg.totalElemCnt`）计算Tile整块数量与尾块元素个数，申请输入、输出、临时变量映射的`AscendC::LocalTensor`的UB空间，循环处理`Tile`整块元素，一次性处理`Tile`尾块元素。
```cpp
template <typename ArgTup>
__aicore__ inline void Run(Arguments& cfg, ArgTup& argTuple) {
    // 动态计算单个block需要处理的tile整块数量
    cfg.wholeLoop = cfg.totalElemCnt / BASIC_BLOCK;
    // 动态计算单个block需要处理tile尾块中元素个数
    cfg.tileCnt = cfg.totalElemCnt % BASIC_BLOCK;
    // 调用私有Process函数进行Tile整、尾块的CopyIn、Evaluate、CopyOut过程
    Process(cfg, argTuple);
    // bufPool的后处理
    ...
}
```
#### 3.3.1.3.2 MakeBlockParam接口
`BaseBlockSchedule`提供`MakeBlockParam()`接口，参数和函数说明如下：
- 参数说明：用户输入`Schedule`层承载切分信息的`blockParam`。
- 函数功能说明：`Schedule`层计算切分参数后并填充`blockParam`。
```cpp
static constexpr bool MakeBlockParam(Arguments& blockParam)
{
    blockParam.wholeLoop = ELEMENT_COUNT_IN_TENSOR / BASIC_BLOCK;
    blockParam.tileCnt = ELEMENT_COUNT_IN_TENSOR % BASIC_BLOCK;
    blockParam.basicNum = ELEMENT_COUNT_IN_TENSOR;
    blockParam.ubAssign = {IN_PARAMS_COUNT, OUT_PARAMS_COUNT, LOCAL_VAR_COUNT, ELEMENT_COUNT_IN_TENSOR};
    return true;
}
```
## 3.4 配置Kernel层模板
Kernel层承担来自Device层的分配的计算任务，需对任务进行分解并调度至相应的Block执行。  
当前实现了`KernelBuilder`模板，其具备对逐元素操作进行高效多核任务分配与调度的能力。
### 3.4.1 KernelBuilder模板

```cpp
template <typename BlockOp, const auto &Policy = kernelPolicyDefault, typename ScheduleCfg = KernelConfig,
    class Schedule = Kernel::DefaultSchedule<BlockOp, Policy, ScheduleCfg>>
class KernelBuilder {
public:
    // 调度策略模板类
    using ScheduleClz = Schedule;
    //Kernel层执行函数 调用对应调度策略的执行函数
    template <typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
       Schedule schedule;
       schedule.Run(cfg, args...);
    }
};
```
#### 3.4.1.1 KernelBuilder模板的范式表达
支持用户自定义修改模板参数实现，实现局部拓展，也支持用户按照规则自定义模板类，实现模板拓展。
`KernelBuilder`类定义如下：
```cpp
template <typename BlockOp, const auto &Policy = kernelPolicyDefault, typename ScheduleCfg = KernelConfig,
    class Schedule = Kernel::DefaultSchedule<BlockOp, Policy, ScheduleCfg>>
class KernelBuilder {
public:
    ...
};
```
KernelBuilder模板对外提供四部分内容：
- 对外提供的模板参数信息  

| 参数名        | 描述                          |  
|------------|-----------------------------|
| [BlockOp](#33-配置block层模板)   | Block层模板类                 | 
| [Policy](#34122-policy)   | 采用的切分策略                 |
| [ScheduleCfg](#34123-schedulecfg)   | 根据切分策略计算的切分信息           |
| [Schedule](#34124-schedule)   | 采用不同的切分策略进行调度计算  | 
- 对外提供的静态参数信息 
```cpp 
using ScheduleClz = Schedule; // 提供调度策略供给上层调用
```
- 默认模板构造函数
```cpp
KernelBuilder()
```
- 对外运行接口`Run`
```cpp
__aicore__ inline void Run(ScheduleCfg& cfg, ArgTup& argTuple);
```
#### 3.4.1.2 相关参数说明
#### 3.4.1.2.1 BlockOp
`BlockOp`：Kernel层需要封装的Block层的模板类，并且Block模板类必须指定Block层使用的计算表达式模板。
#### 3.4.1.2.2 Policy
`Policy`：配置的是用户采用的静态切分策略，比如均匀切分和整尾块切分，默认是均匀切分策略。  
当前`KernelBuilder`提供默认的配置策略`KernelPolicy`，结构体定义如下：
```cpp
enum class KernelPolicySegment {
    Auto = 0U,      // 自动切分
    UniformSegment, // 均匀切分
    FullAddTail     // 整块+尾块
};

struct KernelPolicy {
    uint32_t blockDimMax;                 // 使用的最大核数
    KernelPolicySegment segmentPolicy;    // 多核切分策略
};
```
参数说明：  
+ `blockDimMax`： 最大可用的核数。  
+ `segmentPolicy`：多核切分策略，通过枚举值代表不同的切分策略即`KernelPolicySegment`。  
#### 3.4.1.2.3 ScheduleCfg
 `ScheduleCfg`：用户根据不同的切分策略计算出来的切分参数信息。
#### 3.4.1.2.4 Schedule
`Schedule`是计算任务的调度核心，它调用对应的切分策略来执行计算。在扩展性方面，用户可以通过继承`BaseKernelSchedule`基类可实现自定义策略；系统默认的执行策略为 `DefaultSchedule`。  
切分计算整体流程如下：
1. 根据`ScheduleCfg`获取到Kernel层切分参数信息`KernelConfig`。
2. `PrepareParams`函数根据当前核的`BlockId`得到当前`Kernel`需要处理的所有`GlobalTensor`的数据。
3. 根据切分策略计算当前核实际处理的数据量`actualNum`。
4. 调用Block层完成计算。
#### 3.4.1.2.5 动态Param信息
```cpp
struct KernelConfig {             // Kernel 层的切分信息
    uint32_t blockNum = 1;        // 启动的核数
    uint32_t unitNumPerCore = 0;  // 平均每个核一定会处理的单元块个数
    uint32_t moreUnitCoreNum = 0; // 额外需要处理一个整块的核数
    uint32_t tailNum = 0;         // 最后一个核要处理的尾部元素数量
    uint32_t unitNum = 1;         // 单元块元素个数
};
```    
Kernel层提供了结构体`KernelConfig`，用于记录切分信息，内置参数说明：    
+ `blockNum`：计算需要启动的核的数量。   
+ `unitNumPerCore`：平均每个核一定会处理的单元块个数。
+ `moreUnitCoreNum`：额外需要处理一个整块的核数。
+ `tailNum`：最后一个核要处理的尾部元素数量。
+ `unitNum`：一个单元块元素个数。
#### 3.4.1.3 模板对外接口说明
计算逻辑的实际执行是通过`BaseKernelSchedule`类中的`Run`方法实现。接下来对用户可感知的`BaseKernelSchedule`类中的核心函数进行介绍。
##### 3.4.1.3.1 Run接口
`BaseKernelSchedule`提供`Run()`接口，参数和函数说明如下：
1. 参数说明：接收切分计算参数`cfg`，以及Device层传入的输入输出的GM的地址指针`args`。
2. 函数功能说明：接收Device层传入的`GM`指针，根据当前`BlockId`计算需要处理的GM首地址，调用Block层完成计算。
```cpp
template <typename OpParam, typename... Args>
__aicore__ inline void Run(OpParam& cfg, Args... args)
{
    ...
    // 计算当前核实际需要处理的元素总数量
    uint32_t actualNum = CalCurCoreEleCnt(cfg.kernelParam);
    ...
    // 计算当前核需要处理的GM的首地址
    auto params = PrepareParams<Params>(cfg.kernelParam, argTuple);
    ...
    // 调用Block层完成计算
}

```
##### 3.4.1.3.2 MakeKernelParam接口
`BaseKernelSchedule`提供`MakeKernelParam()`接口，参数和函数说明如下：
1. 参数说明：用户输入的向量的Shape信息`shapeInfo`，以及Kernel层承载切分信息的`kernelParam`。
2. 函数功能说明：接收用户输入的向量的Shape信息，计算切分参数并填充`kernelParam`。
```cpp
static bool MakeKernelParam(std::vector<uint32_t> &shapeInfo, ScheduleCfg &kernelParam)
{
    // 计算需要处理的总元素个数
    ···
    // 计算初始的分核基线
    uint32_t basicCoreEleNum = (Policy.minTileNumPerCore * BASIC_BLOCK + actualNAssign - 1) / actualNAssign * actualNAssign; 
    ...
    // 计算切分信息并填充到kernelParam中
}
```


#### 3.4.1.4 KernelBuilder模板配置
用户自定义的样例举例：
```cpp
using KernelOp = Atvoss::EleWise::KernelBuilder<BlockOp, kernelPolicy, Atvoss::EleWise::KernelConfig>;
using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
```
约束说明：  
必须先定义好Block模板类之后再定义Kernel层的模板类。

## 3.5 配置DeviceAdapter模板
Device层提供统一的Adapter模板，不区分模板类型，通过统一的范式表达和使用。
### 3.5.1 DeviceAdapter范式表达
class `DeviceAdapter` 需要一个Kernel层模板作为模板参数，类定义如下：
```cpp
template <typename KernelOp>
class DeviceAdapter{
    ... // 类实现
}
```
`DeviceAdapter`模板对外提供两部分内容：

- 对外提供的静态参数信息
```cpp
using ExprMaker = typename KernelOp::ScheduleClz::ExprMaker;                // 从KernelOp中萃取出来的表达式模板
using BlockOp = typename KernelOp::ScheduleClz::BlockTemplate;                // 从KernelOp中萃取出来的Block层模板
using KernelParamStruct = typename KernelOp::ScheduleClz::ParamStruct;                    // 萃取出Kennel层Arguments类型
using BlockParamStruct = typename KernelOp::ScheduleClz::BlockTemplate::ScheduleClz::ParamStruct // 萃取出Block层Arguments类型
```

- 对外运行接口
接口固定命名`Run()`，它接收通过 `ArgumentsBuilder` 构建的参数对象，执行相应的计算任务。
```cpp
template <typename Args>
int64_t Run(const Args& arguments)
```
参数构建涉及关键组件如下：

- Tensor 构建(Atvoss::Tensor)

  `Atvoss::Tensor`用于封装内存数据，表示多维张量。

  ```cpp
  template<size_t N>
  Tensor(T* dataPtr, uint32_t (&inputShape)[N]) : dataPtr_(dataPtr)
  ```

  - `dataPtr`: 指向实际数据的指针
  - `inputShape`: 定义张量形状的定长数组，最多支持8维

  示例：

  ```cpp
  // 数据准备
  std::vector<float> v1(32*32, 1.0F); // 32x32 的矩阵，填充 1.0
  uint32_t shape[2] = {32, 32};       // 定义形状为 32 行 32 列
  
  // 创建 Tensor 对象
  Atvoss::Tensor<float> t1(v1.data(), shape);
  ```

- ArgumentsBuilder 构建

  `ArgumentsBuilder` 是一个流式构建器，用于配置算子的输入、输出以及Attr属性入参。

  主要方法：

  - `.input(tensor1, tensor2, ...)` 设置算子的输入张量，可接受一个或多个`Atvoss::Tensor`对象。
  - `.output(tensor1, tensor2, ...)` 设置算子的输出张量，可接受一个或多个`Atvoss::Tensor`对象。
  - `.attr("key", value)` 设置算子的Attr属性入参，可接受key-value键值对的形式，value不限类型，可调用多次，配置多个Attr属性入参。
  - `.build()` 生成最终的 arguments 对象。
  
ArgumentsBuilder 构建输入`.input()`、输出`.output()`、Attr`.attr()`没有先后顺序要求，只需要保证构建arguments最后调用`.build()`即可。
  
  示例：

  ```cpp
  auto arguments = Atvoss::ArgumentsBuilder{}
      .input(t1, t2)						// 设置输入张量
      .output(t3)							// 设置输出张量
      .attr("dim", 4)						// 设置Attr属性
      .attr("scale", 0.5f)
      .attr("format", std::string("NCHW"))
      .attr("perm", std::vector<int>{0,2,1,3})
      .attr("keepDim", false)
      .build();							// 构建最终参数对象
  
  auto& inputs = std::get<0>(arguments);	// 获取输入
  auto& outputs = std::get<1>(arguments);	// 获取输出
  auto& attrs = std::get<2>(arguments);	// 获取Attr
  ```

### 3.5.2 使用DeviceAdaptor
- 使用`KernelOp`作为模板参数封装DeviceAdapter模板类
```cpp
using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
```

- 调用DeviceAdapter的运行接口
```cpp
// 设置输入/输出数据
std::vector<float> v1(32*32, 1.0F);
std::vector<float> v2(32*32, 2.0F);
std::vector<float> v3(32*32);

// 设置shape大小
uint32_t shape[2] = {32, 32};

// 构造输入/输出Tensor
Atvoss::Tensor<float> t1(v1.data(), shape);
Atvoss::Tensor<float> t2(v2.data(), shape);
Atvoss::Tensor<float> t3(v3.data(), shape);

// 构造入参信息
auto arguments = Atvoss::ArgumentsBuilder{}
    .input(t1, t2)
    .output(t3)
    .attr("dim", 4)
    .attr("keepDim", false)
    .build();

// 运行
DeviceOp deviceOp;
deviceOp.Run(arguments);
```

### 3.5.3 DeviceAdaptor运行接口说明
DeviceAdaptor运行接口内部实现用户无需感知，此处只作简单说明便于用户理解。
DeviceAdaptor运行接口自动完成输入的Host侧参数到Device侧的传递，使用`<<<>>>`发起KernelLaunch，调用Kernel层的运行接口，完成算子计算。运行接口主要流程如下：

- 初始化Acl资源
- 调用Kernel层及Block层的动态参数计算接口获取动态参数
- Host侧到Device侧的数据搬运
- 使用`<<<>>>`完成KernelLaunch，发起对Kernel层的调用
- Device侧到Host侧的结果搬运
- 释放Acl资源

# 4. 基于ATVOSS扩展模板
## 4.1 实现自定义Tile层API
### 4.1.1 定义表达式
#### 4.1.1.1 表达式定义
在[math.h](../include/utils/expression/math.h)（基础函数）和[tile_evaluator_transcendental.h](../include/tile/tile_evaluator_transcendental.h)（超越函数）文件进行表达式的定义。
```cpp
DeclareUnaryOp(Name); // Name为需要定义的一元表达式，例如Sqrt
DeclareBinaryOp(Name); // Name为需要定义的二元表达式，例如Max
```
### 4.1.2 接口定义
基础函数在[tile_evaluator_math.h](../include/tile/tile_evaluator_math.h)，超越函数在[tile_evaluator_transcendental.h](../include/tile/tile_evaluator_transcendental.h)，
添加对新表达式的处理，如下格式：
```cpp
template<typename T, typename U, typename V>
struct Evaluator<OpAssign<T, OpAdd<U, V>>> {
    using Type = void;

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAssign<T, OpAdd<U, V>> &op,
                               ArgTup &args,
                               LocalVarTup &localVars,
                               Arguments&... arguments) const {
        using Dtype = Dtype_t<T>;
        uint32_t count = getShape<T, Operation::Unary>(arguments...);
        return AddAssign<Dtype>(
                Evaluator<T>{}(op.GetLhs(), args, localVars),
                Evaluator<U>{}(op.GetRhs().GetLhs(), args, localVars),
                Evaluator<V>{}(op.GetRhs().GetRhs(), args, localVars),
                count);
    }
};
```
### 4.1.3 AscendC接口调用
在[tile_ascendc_math.h](../include/tile/tile_ascendc_math.h)（基础函数）和[tile_ascendc_transcendental.h](../include/tile/tile_ascendc_transcendental.h)（超越函数）中，
新增对应的`Assign`函数，调用`AscendC`的接口。
```cpp
template <typename OperationShape, typename T>
__aicore__ inline void AddAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0,
                                 const AscendC::LocalTensor<T>& src1, OperationShape& operationShape)
{
    AscendC::Add(dst, src0, src1, operationShape.axis0);
}
```