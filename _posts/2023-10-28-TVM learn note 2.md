---
title: TVM Learning Note 2
author: xuao
date: 2023-10-28 17:27:00 +800
categories: [TVM]
tags: [TVM, Schedule, GPU, TE]
---

# TVM 学习笔记：进阶

## 1 TVM 原理

TVM 是一个开源的深度学习编译器，适用于 CPU、GPU、ARM 等多种硬件架构

![tvm_image2]({{ site.url }}/my_img/TVM_image13.png)

1. **从 TensorFlow、PyTorch 或 ONNX 等框架导入模型**

2. **翻译成 TVM 的高级模型语言 Relay**

   Realy 是神经网络的功能语言和中间表示（IR），应用图级优化 pass 来优化模型

3. **降级为张量表达式（TE）表示**

   降级是指将较高级的表示转换为较低级的表示

   应用了高级优化之后，Relay 通过运行 FuseOps pass，把模型划分为许多小的子图，并将子图降级为 TE 表示

   为将 Relay 表示转换为 TE 表示，TVM 包含了一个张量算子清单（TOPI）

4. **使用 auto-tuning 模块 AutoTVM 或 AutoScheduler 搜索最佳 schedule**

   schedule 为 TE 中定义的算子或子图指定底层循环优化

   TVM 中有两个 auto-tuning 模块：

   + AutoTVM：**基于模板**的 auto-tuning 模块，运行搜索算法以在用户定义的模板中找到可调 knob 的最佳值
   + AutoScheduler（又名 Ansor）：**无模板**的 auto-tuning 模块，通过分析计算定义自动生成搜索空间，在其中搜索最佳 schedule

5. **为模型编译选择最佳配置**

   为每个子图选择最佳 schedule

6. **降级为张量中间表示**（TIR，TVM 的底层中间表示）

   所有 TE 子图降级为 TIR 并通过底层优化 pass 进行优化

   优化的 TIR 降级为硬件平台的目标编译器

   TVM 支持多种不同的编译器后端

7. **编译成机器码**

   TVM 可将模型编译为可链接对象模块，然后轻量级 TVM runtime 可以用 C 语言的 API 来动态加载模型，也可以为 Python 和 Rust 等其他语言提供入口点

   或将 runtime 和模型放在同一个 package 里时，TVM 可以对其构建捆绑部署



## 2 使用张量表达式处理算子

张量表达式（TE），TE 用**纯函数式语言**描述张量积算，即每个函数表达式都不会产生副作用（side effect）

从 TVM 的整体来看，Relay 将计算描述为**一组算子**，每个算子都可以表示为一个 TE 表达式，其中每个 TE 表达式接收输入张量并产生一个输出张量。

TVM 使用特定领域的张量表达式来进行有效的内核构建。

学习使用 TE 和调度原语来进行优化。

### 2.1 在 TE 中为 CPU 编写并调度向量加法

```python
import tvm
import tvm.testing
from tvm import te
import numpy as np

target = tvm.target.Target(target="llvm -mcpu=tigerlake", host="llvm -mcpu=tigerlake")
```

#### 2.1.1 描述张量计算

```python
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
```

#### 2.1.2 为计算创建默认 schedule

TVM 要求用户提供 schedule，这是对如何执行计算的描述。

创建一个按行迭代计算 C 的 schedule：

```python
s = te.create_schedule(C.op)
```

#### 2.1.3 编译

用 TE 表达式和 schedule 可为目标语言和架构生成可运行的代码。

TVM 提供 **schedule**、**schedule 中的 TE 表达式列表**、**target** 和 **host**，以及**正在生成的函数名**。

输出结果时一个类型擦除函数（type-erased function），可直接从 Python 调用。

```python
fadd = tvm.build(s, [A, B, C], target, name="myadd")
```

#### 2.1.4 运行

编译后的 TVM 函数提供了一个任何语言都可调用的 C API。

首先创建 TVM 编译调度的目标设备，然后初始化设备中的张量，并执行自定义的加法操作。

```python
dev = tvm.device(tgt.kind.name, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

fadd(a, b, c)
```

#### 2.1.5 更新 schedule 以使用并行性

了解调度 schedule 的作用，使用它们来**优化不同架构的张量表达式**

schedule 是用多种方式来转换表达式的一系列步骤。

默认调度的张量加法是串行的，可以实现为在所有处理器线程上并行化：

```python
s = te.create_schedule(C.op)
s[C].parallel(C.op.axis[0])
```

#### 2.1.6 更新 schedule 以使用向量化

现代 CPU 可以对浮点执行 SIMD 操作，利用这一优势，可将另一个调度应用于计算表达式中。

首先，用拆分调度原语将调度拆分为内部循环和外部循环。

内部循环可用向量化调度原语来调用 SIMD 指令，然后可用并行调度原语对外部循环进行并行化。

选择拆分因子作为 CPU 上的线程数量。

```python
s = te.create_schedule(C.op)

factor = 4	# 和 CPU 的线程数量匹配

outer, inner = s[C].split(C.op.axis[0], factor=factor)

s[C].parallel(outer)
s[C].vectorize(inner)

fadd_vector = tvm.build(s, [A, B, C], tgt, name="myadd_vector")
```

#### 2.1.7 在 GPU 上运行

修改 target：

```python
target = tvm.target.Target(target="cuda", host="llvm")
```

修改 schedule：

```python
s = te.create_schedule(C.op)

bx, tx = s[C].split(C.op.axis[0], factor=64)

# 最终必须将迭代轴 bx 和 tx 和 GPU 计算网格绑定。

s[C].bind(bx, te.thread_axis("blockIdx.x"))
s[C].bind(tx, te.thread_axis("threadIdx.x"))
```

### 2.2 保存和加载已编译的模块

[使用张量表达式处理算子 | Apache TVM 中文站 (hyper.ai)](https://tvm.hyper.ai/docs/tutorial/tensor_expr)

### 2.3 TE 调度原语

- split：将指定的轴按定义的因子拆分为两个轴。
- tile：通过定义的因子将计算拆分到两个轴上。
- fuse：将一个计算的两个连续轴融合。
- reorder：可以将计算的轴重新排序为定义的顺序。
- bind：可以将计算绑定到特定线程，在 GPU 编程中很有用。
- compute_at：TVM 默认将在函数的最外层或根部计算张量。 compute_at 指定应该在另一个算子的第一个计算轴上计算一个张量。
- compute_inline：当标记为 inline 时，计算将被扩展，然后插入到需要张量的地址中。
- compute_root：将计算移动到函数的最外层或根部。这意味着当前阶段的计算完全完成后才可进入下一阶段。

### 2.4 使用 TE 手动优化矩阵乘法

~~TOBEDONE~~





