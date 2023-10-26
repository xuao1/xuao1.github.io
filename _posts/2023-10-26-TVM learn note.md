---
title: TVM Learning Note
author: xuao
date: 2023-10-26 15:33:00 +800
categories: [TVM]
tags: [TVM, Compilor]
---

# TVM 学习笔记

## 1 TVM 原理

TVM 是一个开源的深度学习编译器，适用于 CPU、GPU、ARM 等多种硬件架构

![tvm_image1]({{ site.url }}/my_img/TVM_image1.png)

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

   + AutoTVM：基于模板的 auto-tuning 模块，运行搜索算法以在用户定义的模板中找到可调 knob 的最佳值
   + AutoScheduler（又名 Ansor）：无模板的 auto-tuning 模块，通过分析计算定义自动生成搜索空间，在其中搜索最佳 schedule

5. **为模型编译选择最佳配置**

   为每个子图选择最佳 schedule

6. **降级为张量中间表示**（TIR，TVM 的底层中间表示）

   所有 TE 子图降级为 TIR 并通过底层优化 pass 进行优化

   优化的 TIR 降级为硬件平台的目标编译器

   TVM 支持多种不同的编译器后端

7. **编译成机器码**

   TVM 可将模型编译为可链接对象模块，然后轻量级 TVM runtime 可以用 C 语言的 API 来动态加载模型，也可以为 Python 和 Rust 等其他语言提供入口点

   或将 runtime 和模型放在同一个 package 里时，TVM 可以对其构建捆绑部署

## 2 从源码安装 TVM

![tvm_image2]({{ site.url }}/my_img/TVM_image2.png)

构建 cpptest:

![tvm_image2]({{ site.url }}/my_img/TVM_image3.png)

运行 cpptest：

![tvm_image2]({{ site.url }}/my_img/TVM_image4.png)

![tvm_image2]({{ site.url }}/my_img/TVM_image5.png)

## 3 使用 TVMC 编译和优化模型

TVMC，是 TVM 的命令行驱动程序，执行 TVM 功能（包括对模型的自动调优、编译、分析和执行）

### 3.1 使用 TVMC

TVMC 是 Python 应用程序，用 Python 包安装 TVM 时，会得到一个叫 tvmc 的命令行应用程序

![tvm_image2]({{ site.url }}/my_img/TVM_image6.png)

### 3.2 获取模型

使用 ResNet-50 v2. ResNet-50 是一个用来对图像进行分类的 50 层深的卷积神经网络

本次使用 ONNX 格式的模型

TVMC 支持用 Keras、ONNX、TensorFlow、TFLite 和 Torch 创建的模型。可用 `--model-format` 选项指明正在使用的模型格式

### 3.3 将 ONNX 模型编译到 TVM Runtime

```shell
tvmc compile \
--target "llvm" \
--input-shapes "data:[1,3,224,224]" \
--output resnet50-v2-7-tvm.tar \
resnet50-v2-7.onnx
```

编译的输出结果是模型的 TAR 包，即为**模块（module）**

用以下命令查看 tvmc compile 在模块（module）中创建的文件：

```shel
mkdir model
tar -xvf resnet50-v2-7-tvm.tar -C model
ls model
```

![tvm_image2]({{ site.url }}/my_img/TVM_image7.png)

- `mod.so` 是可被 TVM runtime 加载的模型，表示为 C++ 库。
- `mod.json` 是 TVM Relay 计算图的文本表示。
- `mod.params` 是包含预训练模型参数的文件。

### 3.4 使用 TVMC 运行来自编译模块的模型

将模型编译到模块（module）后，可用 TVM runtime 对其进行预测

TVMC 具有内置的 TVM runtime，可以运行已编译的 TVM 模型，运行前还需要模型的有效输入。

模型的张量 shape、格式和数据类型各不相同，所以，大多数模型都需要预处理和后处理

TVMC 采用了 NumPy 的 .npz 格式的输入和输出

#### 3.4.1 输入预处理

RetNet-50 v2 的输入应该是 ImageNet 格式，用一个脚本获得 imagenet_cat.npz

#### 3.4.2 运行编译模块

有了模型和输入数据，接下来运行 TVMC 进行预测：

```shell
tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz \
resnet50-v2-7-tvm.tar
```

运行后会输出一个新文件 predictions.npz，其中包含 NumPy 格式的模型的输出张量

#### 3.4.3 输出后处理

从编译模块的输出中提取标签，用一个脚本

![tvm_image2]({{ site.url }}/my_img/TVM_image8.png)

### 3.4 自动调优 ResNet 模型

以前的模型被编译到 TVM runtime 上运行，因此不包含特定于平台的优化

接下来将使用 TVMC，针对工作平台构建优化模型

TVM 中的调优是指，**在给定 target 上优化模型**，使其运行得更快。与训练或微调不同，它不会影响模型的准确性，而只会影响 runtime 性能。

TVM 实现并运行许多不同算子的变体，以查看哪个性能最佳，运行的结果存储在调优记录文件中，即 tune 命令的最终输出中。

默认搜索算法需要 xgboost：`pip install xgboost`

在 tune 命令中，为 `--target` 指定更具体的 target 时，会得到更好的结果。

TVMC 针对模型的参数空间进行搜索，为算子尝试不同的配置，然后选择平台上运行最快的配置。虽然这是基于 CPU 和模型操作的引导式搜索，但仍需要几个小时才能完成搜索

> The given `--target` flag in TVM for Intel processors like yours could be `llvm -mcpu=tigerlake` reflecting the architecture of your 11th Gen Intel Core i5-11300H.

> 疑惑：怎么确定适合的 mcpu，按照上述 `tigerlake`，运行效果并不好

```shell
tvmc tune \
--target="llvm -mcpu=broadwell"\
--trials 20000 \
--output resnet50-v2-7-autotuner_records.json \
resnet50-v2-7.onnx
```

输入为原模型，输出为调优记录文件

![tvm_image2]({{ site.url }}/my_img/TVM_image9.png)

### 3.5 使用调优数据编译优化模型

使用上述 tune 的输出作为参数，进行编译：

```shell
tvmc compile \
--target "llvm" \
--input-shapes "data:[1,3,224,224]" \
--tuning-records resnet50-v2-7-autotuner_records.json  \
--output resnet50-v2-7-tvm_autotuned.tar \
resnet50-v2-7.onnx
```

经过验证，优化模型会产生几乎相同的预测结果

### 3.6 比较调优和未调优的模型

各运行 100 次，统计运行时间（上面是未调优，下面是调优后）：

![tvm_image2]({{ site.url }}/my_img/TVM_image10.png)

## 4 使用 TVMC Python 入门：TVM 的高级 API

```shell
mkdir myscripts
cd myscripts
wget https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx
mv resnet50-v2-7.onnx my_model.onnx
touch tvmcpythonintro.py
```

下载 resnet 模型，创建运行脚本 Python 文件 `tvmcpythonintor.py`，接下来编辑该 Python 文件.

### 4.0 导入

```python
from tvm.driver import tvmc
```

### 4.1 加载模型

将模型导入 TVMC。即将模型从支持的框架，转换为 TVM 的高级图形表示语言 Relay.

目前支持的框架：Keras、ONNX、TensorFlow、TFLite 和 PyTorch

```python
model = tvmc.load('my_model.onnx')
```

所有框架都支持用 shape_dict 参数覆盖输入 shape。对于大多数框架，这是可选的；但对 PyTorch 是必需的，因为 TVM 无法自动搜索它

### 4.2 编译

模型已经转换为用 Relay 表示，下一步是将其编译到要运行的硬件（target），即翻译成目标机器可理解的底层语言

编译模型需要一个 tvm.target 字符串。

1. cuda (英伟达 GPU)
2. llvm (CPU)
3. llvm -mcpu=cascadelake（英特尔 CPU）

```python
package = tvmc.compile(model, target="llvm") 
```

编译完成后返回一个 package。

### 4.3 运行

编译后的 package 可在目标硬件上运行

设备输入选项有：CPU、Cuda、CL、Metal 和 Vulkan。

处理输出输出数据时，需要 numpy 库。

```python
input_data = np.load("imagenet_cat.npz")
result = tvmc.run(package, device="cpu", inputs=input_data)
np.savez("predictions.npz", output_0=result)
```

### 4.4 未调优运行的完整代码

```python
from tvm.driver import tvmc
import numpy as np

# 加载模型
model = tvmc.load("my_model.onnx")

# 编译模型
package = tvmc.compile(model, target="llvm")    # 未使用调优

# 运行模型
input_data = np.load("imagenet_cat.npz")
result = tvmc.run(package, device="cpu", inputs=input_data)
np.savez("predictions.npz", output_0=result.outputs['output_0'])
```

### 4.5 调优

在编译前可以通过调优来提高最终模型的运行速度

```python
tvmc.tune(model, target="llvm")
```

可以使用 Autoscheduler：

```python
tvmc.tune(model, target="llvm", enable_autoscheduler = True)
```

保存调优结果，以便后续使用：

```python
log_file = "records.json"
tvmc.tune(model, target="llvm", tuning_records=log_file)
tvmc.tune(model, target="llvm", prior_records=log_file)	# 会使用先前的 tuning 结果，并在此基础上继续优化
```

应用调优结果编译：

```python
package = tvmc.compile(model, target="llvm", tuning_records="records.json")
```

### 4.5 调优运行的完整代码

```python
from tvm.driver import tvmc
import numpy as np

# 加载模型
model = tvmc.load("my_model.onnx")

# 调优
log_file = "records.json"
tvmc.tune(model, target="llvm -mcpu=broadwell", enable_autoscheduler = True, tuning_records=log_file)

# 编译模型
package = tvmc.compile(model, target="llvm", tuning_records=log_file) # 使用调优

# 运行模型
input_data = np.load("imagenet_cat.npz")
result = tvmc.run(package, device="cpu", inputs=input_data)
np.savez("predictions.npz", output_0=result.outputs['output_0'])
```

运行 `python3 tvmcpythonintro.py`：

![tvm_image2]({{ site.url }}/my_img/TVM_image11.png)



## 5 使用 Python 接口（AutoTVM）编译和优化模型

本节使用 TVM 的 Python API 来实现，完成：

+ 使用 TVM runtime 编译模型，运行模型进行预测
+ 使用 TVM 进行调优，使用调优数据重新编译模型，运行优化模型进行预测

### 5.0 导入依赖





















