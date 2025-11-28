# 1. Atvoss概述
Atvoss针对昇腾AI处理器的Vector计算的多核并行计算过程，提供了一套统一的编程模型，可简化用户开发Vector算子的复杂度，简化Tiling计算，并支持模板的自由组合及模板扩展。

编程模型包含如下分层，由高到低分别是：

- Device层：Host侧调用总入口，完成参数校验、Acl资源管理、Host侧与Device侧的数据管理、切分计算、Workspace管理、Kernel调用等逻辑。
- Kernel层：Kernel函数总入口，负责多核间的任务分解，控制Block的调度。
- Block层：负责单核的任务分解到多个Tile块，控制数据搬运/计算数据流进行流水编排处理。
- Tile层：对Ascend C基础API进行封装，提供更大Tile块的搬运、计算等能力。
- Basic层：使用Ascend C基础API能力完成数据搬运计算等基础操作。
<br><img src="./images/architecture.png" width="50%" height="50%" style="margin: 20px 0;"><br>
# 2. 公共基础工具
## 2.1 表达式模板
Atvoss使用表达式模板进行计算流的描述，表达式模板是C++利用运算符重载、类型计算来实现延迟计算的技巧。表达式模板的基本要点是在形成使用运算符或进行函数调用时，我们只是把要进行的操作记录下来不立即进行计算。直到最后Tile层计算的时候，才会进行计算操作。详细的表达式模板实现，可从[Expression表达式模板](../include/utils/expression/expression.h)进入阅读代码。

# 3. 基于Atvoss的算子开发
## 3.1 实现自定义计算表达式
Atvoss模板支持Expr的计算表达，用户通过计算表达完成对自定义算子的描述。具体使用样例如下：
```cpp
static constexpr int32_t HEIGHT = 1;
static constexpr int32_t WIDTH = 32;
using TileShape = Atvoss::Shape<HEIGHT, WIDTH>;

struct RmsNormCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto in1 = Atvoss::PlaceHolder<1, Tensor<DtypeV1>, Atvoss::ParamUsage::in>();
        auto in2 = Atvoss::PlaceHolder<2, Tensor<DtypeV2>, Atvoss::ParamUsage::in>();
        auto out = Atvoss::PlaceHolder<3, Tensor<DtypeV3>, Atvoss::ParamUsage::out>();
        auto temp = Atvoss::PlaceHolderTmpLike<1>(in1);

        return (temp = in1 * in1,
                out = ReduceSum<Atvoss::Patterns::Pattern::AR>(temp),
                out = Broadcast<Atvoss::Patterns::Pattern::AB>(out),
                temp = Divs<WIDTH>(out),
                out = Sqrt(temp),
                temp = in1 / out,
                out = in2 * temp);
    }
};
```

以上代码为例，定义一个`RmsNormOp`的Expr模板类。Tile块的切分设置，使用如下示例的固定写法，使用`Atvoss::Shape`模板定义每个维度的元素大小。
```cpp
using TileShape = Atvoss::Shape<HEIGHT, WIDTH>;
```
`Get()`接口完成算子计算操作的描述，需要先定义输入输出，再定义基于输入输出的计算表达，设置说明如下：

| 表达式元素| 功能描述|约束说明|
| ------------ | ------------ |---|
| `PlaceHolder` | 定义输入输出参数信息，传入的模板参数为<序号，Type，输入输出标识>|序号需要按照`PlaceHolder`定义的先后顺序，从1开始累加； Type固定设置为 `Tensor<T>`， 用户无需感知；输入输出标识为可选参数，默认为`ParamUsage::in`，还可设置为`ParamUsage::out`|
| `PlaceHolderTmpLike` | 定义计算时需要使用的临时buffer|模板参数是标识该临时buffer内存大小跟随哪个`PlaceHolder`的|
| 返回值 | 使用Tile层API描述计算过程|每次使用1个Tile层的API操作实现一次计算过程，每个计算过程是按“,”隔开，用户需要保证计算表达的正确性|

## 3.2 Tile层API
1个Tile层API有两个接口呈现方式，两者一一对应， 用户在计算图中使用API接口名来描述计算操作过程，Tile层内部则是调用实现接口完成实际的计算过程。

- API接口名
  是用户在计算图中调用时使用的接口名，例如`Sqrt()`接口。
```cpp
_2 = Sqrt(_1) // _1 是Sqrt()的输入，_2是输出
```

- 实现接口名
  是Tile层的真正执行计算时调用的接口定义。

### 已支持的Tile层API列表
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
template <typename OperationShape, Atvoss::Patterns::Pattern pattern, typename T>
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
        return (dst = Broadcast<Atvoss::Patterns::Pattern::AB>(src));
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
template <typename OperationShape, Atvoss::Patterns::Pattern pattern, typename T>
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
        return (dst = ReduceSum<Atvoss::Patterns::Pattern::AR>(src));
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
template <typename ExprMaker, const auto& Policy=policyDefault, class OpTraits = void, typename Arguments = Atvoss::Block::Config,
    class Schedule = DefaultSchedule<ExprMaker, Policy, OpTraits, Arguments>>
class BlockBuilder{
    ... // 类实现
}
```
模板参数说明：

| 参数名                             | 描述                |
|---------------------------------|-------------------|
| [ExprMaker](#31-实现自定义计算表达式)     | 自定义计算表达式          |
| [Policy](#33121-Policy)         | Block层UB相关配置项     |
| [Arguments](#33122-Arguments)   | Block层数据元素切分结果配置项 |
| [Schedule](#33123-Schedule（预留）) | Block层切分算法        |

#### 3.3.1.1 BlockBuilder模板的范式表达
`class BlockBuilder`模板对外提供三个部分内容：

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
struct PolicyEleWise {
    uint32_t ubSizeMax = 190 * 1024;  // 最大使用的UB空间大小，默认190K
    Shape tileShape{};               // 一维: 不需要额外对齐， 二维： 按Shape的第二个配置值对齐
};
```
#### 3.3.1.2.2 ScheduleCfg
`ScheduleCfg`：根据不同策略计算出来的`Tile`层处理的元素信息。
```cpp
// Block层的元素切分信息
struct Config {
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
`Schedule`：多Tile切分算法类实际计算。
- 对外提供的静态参数信息。
```cpp
// 算子数学表达式结构体
using ExpressMaker = ExprMaker;
// 算子数学表达式中元素数据类型，类型不同时获取较大的
using MaxSizeType = typename ExpressMaker::maxSizeType;
// device、kernel层感知Block层的配置参数 
using ParamStruct = Arguments;

// 计算表达式类型
using Expr = typename decltype(FakeEXPRESSION)::Type;
// 声明参数类型
using Params = Atvoss::ExprTmpl::Params_t<Expr>; // 萃取表达式所有参数类型列表
using InParams = Atvoss::ExprTmpl::InParams_t<Expr>; // 萃取表达式输入参数类型列表
using OutParams = Atvoss::ExprTmpl::OutParams_t<Expr>; // 萃取表达式输出参数类型列表
using LocalVars = Atvoss::ExprTmpl::LocalVars_t<Expr>; // 萃取表达式临时参数类型列表

// 计算输入，输出，临时参数的数量
static constexpr uint32_t IN_PARAMS_COUNT =  Util::TMP::Size_v<InParams> + 1;
static constexpr uint32_t OUT_PARAMS_COUNT = Util::TMP::Size_v<OutParams> + 1;
static constexpr uint32_t LOCAL_VAR_COUNT =  Util::TMP::Size_v<LocalVars>;

// 计算需要的存活节点数量
static constexpr uint32_t MAX_BUFFER_COUNT = IN_PARAMS_COUNT + OUT_PARAMS_COUNT + LOCAL_VAR_COUNT;
// 单个节点占用UB大小
static constexpr uint32_t UB_TILE_SIZE = Policy.ubSizeMax / MAX_BUFFER_COUNT / 1024 * 1024;
// 内存管理输入变量在UB空间首地址
static constexpr uint64_t UB_ADDR_IN = 0;
// 内存管理输出变量在UB空间首地址
static constexpr uint64_t UB_ADDR_OUT = UB_TILE_SIZE * IN_PARAMS_COUNT;
// 内存管理临时变量在UB空间首地址
static constexpr uint64_t UB_ADDR_CALC = UB_ADDR_OUT + UB_TILE_SIZE * OUT_PARAMS_COUNT;

// 用户配置的layout计算作为单次Tile块元素大小
static constexpr uint32_t BASIC_BLOCK = Atvoss::Tile::GetTotal<0, ExpressMaker>();
```
- 编译期根据[Policy](#33121-Policy)参数配置，计算单个存活节点占用`UB`空间。
```cpp
// 编译期获取单个存活节点占用UB空间大小
static constexpr uint32_t GetEleCntInTensor();
```
- 编译期计算Block层配置，作为默认[ScheduleCfg](#33122-ScheduleCfg)参数
```cpp
// Schedule层默认配置计算
static constexpr bool MakeBlockParam(Arguments &blockParam);
```

- 对`Block`运行接口`Run`
```cpp
// Block层执行函数。
// cfg切分结果配置，通过单Block最大处理元素个数（totalElemCnt）计算Tile整块与尾块个数
// argTuple为GM侧的输入、输出数据空间
template<typename ArgTup>
__aicore__ inline void Run(Arguments &cfg, ArgTup &argTuple);
```

#### 3.3.1.3 模板对外接口说明
#### 3.3.1.3.1 Run接口
`Schedule`提供`Run()`接口按批次完成输入数据从GM->UB，计算，输出结果UB->GM过程，参数和函数说明如下：
- 参数说明：接收切分配置`cfg`，以及`Kernel`层传入的输入、输出GM的地址`argTuple`。
- 函数功能说明：通过Block需要处理的总元素个数（`cfg.totalElemCnt`）计算Tile整块数量与尾块元素个数，申请输入、输出、临时变量映射的`AscendC::LocalTensor`的UB空间，循环处理`Tile`整块元素，一次性处理`Tile`尾块元素。
```cpp
public:
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
`Schedule`提供`MakeBlockParam()`接口，参数和函数说明如下：
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
当前实现了KernelBuilder模板，其具备对逐元素操作进行高效多核任务分配与调度的能力。
### 3.4.1 KernelBuilder模板

```cpp
template <typename BlockOp, const auto &Policy = policyDefault, class OpTraits = void, typename Arguments = Atvoss::Kernel::Config,
    class Schedule = DefaultSchedule<BlockOp, Policy, OpTraits, Arguments>>
class KernelBuilder {
public:
    // 从BlockOp中萃取出来的表达式模板
    using ExprMaker = typename BlockOp::ExpressMaker;
    // 存储切分参数
    using ParamStruct = Arguments;
    // 用户定义的计算表达式的数据类型
    using MaxSizeType = typename ExprMaker::maxSizeType;
    static constexpr auto FakeEXPRESSION = ExprMaker{}.template Get<AscendC::GlobalTensor>();
    // Kernel层计算表达式类型
    using Expr = typename decltype(FakeEXPRESSION)::Type;
    // 萃取表达式所有参数类型列表
    using Params = Atvoss::ExprTmpl::Params_t<Expr>;
    // 从Block层萃取的Tile块大小
    static constexpr uint32_t BASIC_BLOCK = BlockOp::BASIC_BLOCK;
    ...
    // 构造函数
    __aicore__ inline KernelBuilder(){
        ...
    };
    //计算切分参数
    static bool MakeKernelParam(std::vector<uint32_t> &shapeInfo, Arguments &kernelParam){
        ...
    }
    template <typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
        // 根据切分策略循环调用Block完成计算
        ...
    }
}
```
#### 3.4.1.1 KernelBuilder模板的范式表达
支持用户自定义修改模板参数实现，实现局部拓展，也支持用户按照规则自定义模板类，实现模板拓展。
`KernelBuilder`类定义如下：
```cpp
template <typename BlockOp, const auto &Policy = policyDefault, class OpTraits = void, typename Arguments = Atvoss::Kernel::Config, class Schedule = DefaultSchedule<BlockOp, Policy, OpTraits, Arguments>>
class KernelBuilder {
    ...
}
```
KernelBuilder模板对外提供五部分内容：
- 对外提供的模板参数信息  

| 参数名        | 描述                          |  
|------------|-----------------------------|
| [BlockOp](#33-配置block层模板)   | Block层模板类                 | 
| [Policy](#34122-policy)   | 采用的切分策略                 |
| [OpTraits](#34123-optraits预留)   | 当前层输入输出的类型信息                 |
| [Arguments](#34124-arguments)   | 根据切分策略计算的切分信息           |
| [Schedule](#34125-schedule预留)   | 采用不同的切分策略进行调度运行  | 
- 对外提供的静态参数信息 
```cpp 
using ExprMaker = typename BlockOp::ExpressMaker;               // 从BlockOp中萃取出来的表达式模板
using MaxSizeType = typename ExprMaker::maxSizeType;            // 用户定义的计算表达式的数据类型
using Expr = typename decltype(FakeEXPRESSION)::Type;               // Kernel层计算表达式类型
using Params = Atvoss::ExprTmpl::Params_t<Expr>;                 // 萃取表达式所有参数类型列表
static constexpr uint32_t BASIC_BLOCK = BlockOp::BASIC_BLOCK;   // 从Block层萃取的Tile块大小
```
- 默认模板构造函数
```cpp
KernelBuilder()
```
- 计算切分参数函数
```cpp
static bool MakeKernelParam(std::vector<uint32_t> &shapeInfo, Arguments &kernelParam)
```
使用时需要传入Tensor的Shape信息`shapeInfo`，来计算总的元素个数，`kernelParam`为存储切分信息的结构体。
- 对外运行接口，接口固定命名`Run()`，接口接受`OpParam`参数，以及不定长参数`template <typename... Args>`
```cpp
template <typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args);
```
在使用时，限制`Run`的入参`cfg`必须是`Arguments`类型，可变参数`args`必须是GM的地址指针，并且入参需要严格和`ExprMaker`计算表达中描述的算子输入输出信息保持一致（参数个数、顺序）。

#### 3.4.1.2 相关参数说明
#### 3.4.1.2.1 BlockOp
`BlockOp`：Kernel层需要封装的Block层的模板类，并且Block模板类必须指定Block层使用的计算表达式模板。
#### 3.4.1.2.2 Policy
`Policy`：配置的是用户采用的静态切分策略，比如均匀切分和整尾块切分，默认是均匀切分策略。  
当前`KernelBuilder`提供默认的配置策略`PolicyEleWise`，结构体定义如下：
```cpp
enum class PolicySegment
{
    Auto = 0U,      // 自动切分
    UniformSegment, // 均匀切分
    FullAddTail     // 整块+尾块
};

struct PolicyEleWise { 
    uint32_t blockDimMax;       // Maximum blockDim used
    uint32_t tailBlock;         //  =0 只处理整块，=1 包含整尾块，=2 只处理尾块(预留) 
    uint32_t shapeNAssign = 0;  // 是否需要N对齐，0代表不需要做特殊的N对齐操作，注意：N必须32对齐
    uint32_t minTileNumPerCore; // 单核最小的需要分配的Tile块的个数
    PolicySegment segmentPolicy;// 多核切分策略
};
```
参数说明：  
+ `blockDimMax`： 最大可用的核数。  
+ `tailBlock`：整尾块的标识（预留）。
+ `shapeNAssign` ：是否需要特殊的N对齐（用于Reduce操作等），0代表不需要做特殊的N对齐操作，默认32对齐，注意：用户自定义的N必须是32的倍数。
+ `minTileNumPerCore`：单核最小的需要分配的Tile块的个数，用户可使用该参数控制单核处理的元素数量。
+ `segmentPolicy`：多核切分策略，通过枚举值代表不同的切分策略即`PolicySegment`。  
#### 3.4.1.2.3 OpTraits（预留）
`OpTraits`：当前层的输入输出的类型信息。
#### 3.4.1.2.4 Arguments
 `Arguments`：用户根据不同的切分策略计算出来的切分参数信息。
#### 3.4.1.2.5 Schedule（预留）
`Schedule`调用对应切分策略完成计算。切分计算整体流程如下：
1. 根据`Arguments`获取到Kernel层切分参数信息`Config`。
2. `PrepareParams`函数根据当前核的`BlockId`得到当前`Kernel`需要处理的所有`GlobalTensor`的数据。
3. 计算当前核实际处理的数据量`actualNum`。
4. 调用Block层完成计算。
#### 3.4.1.2.6 动态Param信息
```cpp
struct Config {                   // Kernel 层的切分信息
    uint32_t blockNum = 1;        // 启动的核数
    uint32_t unitNumPerCore = 0;  // 平均每个核一定会处理的单元块个数
    uint32_t moreUnitCoreNum = 0; // 额外需要处理一个整块的核数
    uint32_t tailNum = 0;         // 最后一个核要处理的尾部元素数量
    uint32_t unitNum = 1;         // 单元块元素个数
};
```    
Kernel层提供了结构体`Config`，用于记录切分信息，内置参数说明：    
+ `blockNum`：计算需要启动的核的数量。   
+ `unitNumPerCore`：平均每个核一定会处理的单元块个数。
+ `moreUnitCoreNum`：额外需要处理一个整块的核数。
+ `tailNum`：最后一个核要处理的尾部元素数量。
+ `unitNum`：一个单元块元素个数。
#### 3.4.1.3 模板对外接口说明
##### 3.4.1.3.1 Run接口
模板提供`Run()`接口，参数和函数说明如下：
1. 参数说明：接收切分计算参数`cfg`，以及Device层传入的输入输出的GM的地址指针`args`。
2. 函数功能说明：接收Device层传入的`GM`指针，根据当前`BlockId`计算需要处理的GM首地址，调用Block层完成计算。
```cpp
template <typename BlockOp, const auto &Policy = policyDefault, class OpTraits = void, typename Arguments = Atvoss::EleWise::Config,class Schedule = DefaultSchedule<BlockOp, Policy, OpTraits, Arguments>>
class KernelBuilder {
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
        blockOp.Run(configBlock, convertArgs);
    }
};
```
##### 3.4.1.3.2 MakeKernelParam接口
模板提供`MakeKernelParam()`接口，参数和函数说明如下：
1. 参数说明：用户输入的向量的Shape信息`shapeInfo`，以及Kernel层承载切分信息的`kernelParam`。
2. 函数功能说明：接收用户输入的向量的Shape信息，计算切分参数并填充`kernelParam`。
```cpp
static bool MakeKernelParam(std::vector<uint32_t> &shapeInfo, Arguments &kernelParam)
{
        // 计算需要处理的总元素个数
        uint32_t  totalEleNum = 1;
        for(int i = 0;i < shapeInfo.size();i++){
            totalEleNum = totalEleNum * shapeInfo[i];
        }
        
        uint32_t actualNAssign = Policy.shapeNAssign == 0 ? 32 : Policy.shapeNAssign;
        // 初始的分核基线
        uint32_t basicCoreEleNum = (Policy.minTileNumPerCore * BASIC_BLOCK + actualNAssign - 1) / actualNAssign * actualNAssign; 
        ...
        kernelParam.unitNum = actualNAssign;
        if (totalEleNum <= basicCoreEleNum) {
            kernelParam.blockNum = 1;
            kernelParam.tailNum = totalEleNum;
            ...
        } else {
            // 单核可以处理的对齐单元的个数
            uint32_t basicCoreUnitNum = basicCoreEleNum / actualNAssign; 
            // 总共有多少个对齐单元
            uint32_t totalUnitCnt = totalEleNum / actualNAssign;
            // 计算启动核的数量  
            kernelParam.blockNum = (totalUnitCnt + basicCoreUnitNum -1 ) / basicCoreUnitNum;
            if (kernelParam.blockNum > Policy.BlockDimMax) {
                kernelParam.blockNum = Policy.BlockDimMax;
            }
            // 计算每个核处理的对齐单元的个数
            kernelParam.unitNumPerCore = totalUnitCnt / kernelParam.blockNum;
            // 计算额外需要处理整块单元的核数
            kernelParam.moreUnitCoreNum = totalUnitCnt % kernelParam.blockNum; 
            // 计算剩余的总元素数量
            kernelParam.tailNum = totalEleNum % actualNAssign; 
        }
        return true;
}
```

#### 3.4.1.4 模板对外静态信息说明
```cpp
using ExprMaker = typename BlockOp::ExpressMaker;             // 从BlockOp中萃取出来的表达式模板
using MaxSizeType = typename ExprMaker::maxSizeType;          // 用户定义的计算表达式的数据类型
using Expr = typename decltype(FakeEXPRESSION)::Type;             // Kernel层计算表达式类型
using Params = Atvoss::ExprTmpl::Params_t<Expr>;               // 萃取表达式所有参数类型列表
static constexpr uint32_t BASIC_BLOCK = BlockOp::BASIC_BLOCK; // 从Block层萃取的Tile块大小
```
#### 3.4.1.5 KernelBuilder模板配置
用户自定义的样例举例：
```cpp
using KernelOp = Atvoss::EleWise::KernelBuilder<BlockOp, kernelPolicyWidthAssign>;
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
using ExprMaker = typename KernelOp::ExprMaker;     // 从KernelOp中萃取出来的表达式模板
using BlockOp = typename KernelOp::BlockTemplate;   // 从KernelOp中萃取出来的Block层模板
using DType = typename BlockOp::ExpressMaker::type; // 萃取出来的数据类型
using kernelParamStruct = typename KernelOp::ParamStruct;               // 萃取出Kennel层Arguments类型
using blockParamStruct = typename KernelOp::BlockTemplate::ParamStruct; // 萃取出Block层Arguments类型
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
  auto arguments = ATVOSS::ArgumentsBuilder{}
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
using DeviceOp = Atvoss::Device::DeviceAdapter<KernelOp>;
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
template <typename T>
__aicore__ inline void AddAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0,
                                 const AscendC::LocalTensor<T>& src1, uint32_t count = 0)
{
    AscendC::Add(dst, src0, src1, count);
}
```