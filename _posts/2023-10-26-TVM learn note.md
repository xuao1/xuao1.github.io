---
title: TVM Learning Note
author: xuao
date: 2023-10-26 15:33:00 +800
categories: [TVM]
tags: [TVM, Compilor, GPU]
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

包括用于加载和转换模型的 `onnx`、用于下载测试数据的辅助实用程序、用于处理图像数据的 Python 图像库、用于图像数据预处理和后处理的 `numpy`、TVM Relay 框架和 TVM 图形处理器

```python
import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
```

作用分别是：

+ `onnx`：加载和转换深度学习模型
+ `tvm.contrib.download`：TVM 框架提供的下载辅助工具，用于从网络上下载测试数据或模型文件
+ `PIL`：图像处理库，用于加载、处理和保存图像
+ `numpy`：用于科学计算和数值操作的核心库，用于图像数据预处理和后处理
+ `tvm.relay`：TVM 框架中的子模块，用于定义和优化深度学习模型。它提供了一种中间表示形式，允许对模型进行高效的编译和优化
+ `tvm`：TVM 是一个开源的深度学习模型优化和部署框架，支持多种硬件后端
+ `tvm.contrib.graph_executor`：TVM 图形处理器

### 5.1 下载和加载 ONNX 模型

使用 ResNet-50 v2，是一个深度为 50 层的卷积神经网络，适用于图像分类任务

```python
model_url = (
    "https://github.com/onnx/models/raw/main/"
    "vision/classification/resnet/model/"
    "resnet50-v2-7.onnx"
)

model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
onnx_model = onnx.load(model_path)

# 为 numpy 的 RNG 设置 seed，得到一致的结果
np.random.seed(0)
```

### 5.2 下载、预处理和加载测试图像

TVMC 采用了 NumPy 的 `.npz` 格式的输入和输出数据

```python
img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# 重设大小为 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# 输入图像是 HWC 布局，而 ONNX 需要 CHW 输入，所以转换数组
img_data = np.transpose(img_data, (2, 0, 1))

# 根据 ImageNet 输入规范进行归一化
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# 添加 batch 维度，期望 4 维输入：NCHW。
img_data = np.expand_dims(norm_img_data, axis=0)
```

### 5.3 使用 Relay 编译模型

1. 将模型导入到 Relay
2. 用标准优化，将模型构建到 TVM 库中
3. 从库中创建一个 TVM 计算图 runtime 模块

```python
input_name = "data" # 输入名称可能因模型类型而异，可用 Netron 工具检查输入名称
shape_dict = {input_name: img_data.shape}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
```

### 5.4 在 TVM Runtime 执行

```python
# dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
```

### 5.5 收集基本性能数据

现在是**未优化**版本：

```python
import timeit

timing_number = 10
timing_repeat = 10
unoptimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

print(unoptimized)
```

- `timeit.Timer(lambda: module.run())`: 使用lambda函数创建一个无参数的函数，该函数仅执行 `module.run()`。
- `.repeat(repeat=timing_repeat, number=timing_number)`: 对上述lambda函数进行计时。重复计时 `timing_repeat` 次（即10次），每次执行 `timing_number` 次（即10次）`module.run()`。所以之后需要除以 `timing_number`

### 5.6 输出后处理

需要用专为该模型提供的查找表，运行一些后处理

```python
from scipy.special import softmax

# 下载标签列表
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

# 打开输出文件并读取输出张量
scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))
```

运行结果（不包含 5.5 部分）：

![tvm_image2]({{ site.url }}/my_img/TVM_image12.png)

### 5.7 调优模型

在 TVM 的 `autotvm` 模块中，调优是一个重要的步骤，其目标是为给定的任务找到最优的配置以提高性能。调优过程大致分为两个主要阶段：**搜索** 和 **评估**

最简单的调优形式中，需要：target，调优记录文件的存储路径

```python
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
```

设置部分基本参数：

+ `number`：将要测试的不同配置的数量

+ `repeat`：将对每个配置进行多少次测试 

+ `min_repeat_ms`：运行测试需要多长时间，如果重复次数低于此时间，则增加其值

  在 GPU 上进行精确调优时此选项是必需的，在 CPU 调优则不是必需的，将此值设置为 0表示禁

+ `timeout`：每个测试配置运行训练代码的时间上限

```python
number = 10
repeat = 1
min_repeat_ms = 0  # 调优 CPU 时设置为 0
timeout = 10  # 秒

# 创建 autotvm 运行器
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)
```

Runner 负责在硬件上**评估给定配置的性能**，LocalRunner 是运行在本地机器上的一个特定类型的 Runner

```python
tuning_option = {
    "tuner": "xgb",
    "trials": 20,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "resnet-50-v2-autotuning.json",
}
```

+ 使用 XGBoost 算法来指导搜索

+ 试验次数设置为 20，此处这个值比较小。对于 CPU 推荐 1500，对于 GPU 推荐 3000-4000。

+ `early_stopping`：使得搜索提前停止的试验最小值，如果在一系列连续的尝试中没有看到性能改进，参数允许调优过程提前终止

+ measure option 决定了构建试用代码并运行的位置

  定义了两个主要组件：

  + builder 负责从给定配置构建可执行代码
  + runner 负责运行并测量该代码的性能

+ `Tuning_records`：指定将调优数据写入的哪个文件中

```python
# 首先从 onnx 模型中提取任务，mod 是将 onnx 导入到 Relay 时返回的参数
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# 按顺序调优提取的任务
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner_obj = XGBTuner(task, loss_type="rank")
    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )
```

`callbacks` 是函数列表，这些函数在调优过程中的不同时间点被调用。

- `autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix)`

  这个回调函数显示一个进度条，给用户一个直观的了解调优过程的进度。

- `autotvm.callback.log_to_file(tuning_option["tuning_records"])`

  这个回调函数将调优日志写入指定的文件中。

### 5.8 使用调优数据编译优化模型

获取存储在 `resnet-50-v2-autotuning.json`（上述调优过程的输出文件）中的调优记录。编译器会用这个结果，为指定 target 上的模型生成高性能代码。

```python
with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
```

剩余过程一摸一样。





















