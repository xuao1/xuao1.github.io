---
title: Build Gemini
author: xuao
date: 2023-11-09 11:31:00 +800
categories: [Deployment]
tags: [GPU,GPU Sharing,GPU Schedule,Multi-Tenancy,Resource Allocation, Deployment]
---

# 部署测试 Gemini

## 1. System Structure

Gemini 由三部分组成：

+ Scheduler：一个守护进程，管理令牌。基于在资源配置文件（resource-config.txt）中提供的信息，调度器确定要授予令牌的对象。客户端只有在持有有效令牌时才能启动 CUDA 核心。
+ hook library：一个拦截 CUDA 相关函数调用的库。它利用 LD_PRELOAD 机制，强制在任何其他动态链接库之前加载我们的 Hook 库
+ pod manager：一个代理，用于将消息转发给应用程序/调度器。它充当调度器的客户端，每个通过此 Pod 管理器向调度器发送请求的应用程序共享令牌

## 2. Build & Run

启动容器：

```bash
docker run --gpus all --privileged -v $(pwd):/workspace -it --rm tlcpack/ci-gpu:20230504-142417-4d37a0a0 /bin/bash
```

获取项目：

```bash
git clone git@github.com:NTHU-LSALAB/Gemini.git
mv Gemini/ gemini
```

也尝试过：

```bash
wget https://github.com/NTHU-LSALAB/Gemini/archive/refs/tags/v1.1-kubeshare.tar.gz
tar -xzvf v1.1-kubeshare.tar.gz
mv Gemini-1.1-kubeshare/ gemini
```

**这两个的代码不一样。**

**目前决定使用第二种方法。**

### 2.1 Build

```bash
make [CUDA_PATH=/path/to/cuda/installation] [PREFIX=/place/to/install] [DEBUG=1]
```

实际在我的容器内的运行指令为：

```bash
make CUDA_PATH=/usr/local/cuda-11.8 DEBUG=1
```

![Gemini-image1]({{ site.url }}/my_img/Gemini-image1.png)

### 2.2 Run

就是运行 /tool 文件夹下的 Python 脚本

### 2.2.1 launch backend

`launch-backend.py`：启动 scheduler 和 pod manegers

![Gemini-image2]({{ site.url }}/my_img/Gemini-image2.png)

> By default scheduler uses port `50051`, and pod managers use ports starting from `50052` (`50052`, `50053`, ...).

执行命令：

```bash
python3 launch-backend.py ../bin/gem-schd ../bin/gem-pmgr --ip 127.0.0.1 --port 50051 --config ../resource-config.txt
```

![Gemini-image3]({{ site.url }}/my_img/Gemini-image3.png)

### 2.2.2 launch-command

`launch-command.py`：启动 applications

![Gemini-image4]({{ site.url }}/my_img/Gemini-image4.png)

```bash
python3 launch-command.py --name client1 --port 50052 --ip 127.0.0.1 --timeout 10 "/root/test 100"
```

![Gemini-image5]({{ site.url }}/my_img/Gemini-image5.png)

对于该问题，做了以下尝试：

尝试一：

> 尝试将 API hook 编译出来的库添加到 LD_PRELOAD 中：
>
> ```bash
> export LD_PRELOAD=/root/gemini/lib/libgemhook.so.1
> ```
>
> 注：这个命令仅在当前终端有效
>
> 但是似乎并没有什么用

尝试二：

> 注：以下只对 git clone 下来的代码可以尝试（虽然结果也是不行），因为 release 中的 hook.cpp 代码不一样
>
> [__libc_dlopen_mode and __libc_dlsym gone in glibc-2.33.9000+ · Issue #756 · apitrace/apitrace (github.com)](https://github.com/apitrace/apitrace/issues/756)
>
> ![Gemini-image9]({{ site.url }}/my_img/Gemini-image9.png)
>
> 修改 `hook.cpp` 第 74 行：
>
> ```c++
>       (fnDlsym)__libc_dlsym(__libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
> ```
>
> 为：
>
> ```c++
>       (fnDlsym)__libc_dlsym(__libc_dlopen_mode("libc.so", RTLD_LAZY), "dlsym");
> ```
>
> 但是仍然不行

尝试三：

> 尝试将所有的 `__libc_dlopen_mode` 修改为 `dlopen`
>
> 注释掉 extern "C" {void *__libc_dlopen_mode(const char *name, int mode);}
>
> 将之后的所有 `__libc_dlopen_mode` 修改为 `dlopen`
>
> ![Gemini-image6]({{ site.url }}/my_img/Gemini-image6.png)
>
> 所以也类似的修改 `__libc_dlsym` 修改为 `dlsym`
>
> ![Gemini-image7]({{ site.url }}/my_img/Gemini-image7.png)
>
> 出现了新的问题， 通常表示程序在尝试初始化一个对象时发生了递归，而这个对象的构造需要在程序启动时就完成。这有时候会发生在全局对象或静态实例的初始化时，尤其是当这些对象的构造函数中涉及到复杂的逻辑或对其他尚未初始化的全局资源的依赖时。
>
> 改用 git clone 版本的代码：
>
> ![Gemini-image8]({{ site.url }}/my_img/Gemini-image8.png)
>
> 一样的问题

尝试四：

> 尝试 v1.1-bare-metal
>
> ```bash
> wget https://github.com/NTHU-LSALAB/Gemini/archive/refs/tags/v1.1-bare-metal.tar.gz
> tar -xzvf v1.1-bare-metal.tar.gz
> mv Gemini-1.1-bare-metal/ gemini
> ```
>
> 仍然后找不到 `__libc_dlopen_mode` 的问题：
>
> ![Gemini-image10]({{ site.url }}/my_img/Gemini-image10.png)
>
> 重试尝试 3：
>
> ![Gemini-image11]({{ site.url }}/my_img/Gemini-image11.png)

尝试五：

> 基于 v1.1-bare-metal：
>
> 阅读 Makefile：
>
> 由于我使用的是 a100，所以 SMS 应修改为 80
>
> 同时，下载 g++-8：`sudo apt-get install g++-8`
>
> 目前先使用 Ubuntu18.04 的 docker
>
> docker build ：
>
> ```bash
> docker build -t xaubuntu1804 .
> docker run --gpus all --privileged -v $(pwd):/workspace -d xaubuntu1804
> ```
>
> 安装 cuda-11.0:
>
> ```bash
> wget https://developer.download.nvidia.com/compute/cuda/11.0.1/local_installers/cuda_11.0.1_450.36.06_linux.run
> sudo sh cuda_11.0.1_450.36.06_linux.run
> vim ~/.bashrc
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/lib64
> export PATH=$PATH:/usr/local/cuda-11.0/bin
> export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.0
> source ~/.bashrc
> ```
>
> make:
>
> ```
> make CUDA_PATH=/usr/local/cuda-11.0 DEBUG=1
> ```
>
> ![Gemini-image12]({{ site.url }}/my_img/Gemini-image12.png)
>
> 似乎可行，因为没有修改 hook.cpp，也没有报错：
>
> ![Gemini-image13]({{ site.url }}/my_img/Gemini-image13.png)
>
> ![Gemini-image14]({{ site.url }}/my_img/Gemini-image14.png)
>
> 但是莫名其妙不会向下执行。
>
> ~~可能是 hostname 的原因，阅读源码发现，Pod manager 是将 hostname 作为 Pod name，通过在终端输入命令 `hostname`，得到主机的名称：bb409cb7adff。后续阅读代码发现，并不是这回事，是未给出 Pod Name 时才采用 host name~~
>
> 卡住是卡在：执行到 cudaEventCreate 就会卡住
>
> ![Gemini-image15]({{ site.url }}/my_img/Gemini-image15.png)
>
> 之后输出所有拦截到的 cuda 调用，发现最后是卡在：
>
> ![Gemini-image16]({{ site.url }}/my_img/Gemini-image16.png)
>
> 似乎 cuda11.0 不支持这个 api:
>
> [cuda/README.md at master · tmcdonell/cuda (github.com)](https://github.com/tmcdonell/cuda/blob/master/README.md)
>
> ![Gemini-image17]({{ site.url }}/my_img/Gemini-image17.png)
>
> 尝试升级一下 cuda 到 11.7
>
> ```
> wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
> sudo sh cuda_11.7.0_515.43.04_linux.run
> ```
>
> make 时：
>
> ![Gemini-image18]({{ site.url }}/my_img/Gemini-image18.png)
>
> 运行结果：
>
> ![Gemini-image19]({{ site.url }}/my_img/Gemini-image19.png)
>
> 能运行了，但是它拦截了吗？
>
> ![Gemini-image20]({{ site.url }}/my_img/Gemini-image20.png)
>
> 似乎是没拦截到，不确定，但是很奇怪的一点是，initialize() 函数里面的应该有的输出一个也没有
>
> 保留这个容器：3d46283b5e06
>
> 改用 cuda11.1:
>
> ```bash
> wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
> sh cuda_11.1.0_455.23.05_linux.run
> ```
>
> 和 cuda11.0 一样卡住，卡住位置也一样。
>
> 尝试 kubeshare 1.1：
>
> ![Gemini-image21]({{ site.url }}/my_img/Gemini-image21.png)

尝试六：

> 突然发现可以使用实验室集群中的 1080Ti，可以使用与 Gemini 原始配置中的 cuda10，所以新建 Ubuntu18.04 的容器，在其中安装 cuda10.0，尝试运行。
>
> 首先，创建容器，直接使用「尝试五」中的 Dockerfile 即可，然而实际运行不起来，不知道为什么，所以直接 docker pull 了 ubuntu 18:04，手动安装了必要的包，然后安装 cuda10.0：
>
> ```bash
> wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
> ```
>
> 结果是卡在：（仍然是 hook.cpp 中的 cudaEventCreate）
>
> ![Gemini-image22]({{ site.url }}/my_img/Gemini-image22.png)
>
> 再尝试 kubeshare：
>
> ```bash
> wget https://github.com/NTHU-LSALAB/Gemini/archive/refs/tags/v1.1-kubeshare.tar.gz
> ```
>
> ![Gemini-image23]({{ site.url }}/my_img/Gemini-image23.png)
>
> 尝试 cuda11.0：
>
> 与在 a100 上一样的情景:
>
> ![Gemini-image24]({{ site.url }}/my_img/Gemini-image24.png)

尝试七：

> 仍然使用 a100，cuda 11.0，使用自己仓库里面的 gemini 代码（即 git clone gemini 的源码，修改了 Makefile，增加了 hook.cpp 的调试信息）
>
> ```bash
> docker run --gpus all --privileged -v $(pwd):/workspace -d xaubuntu1804
> wget https://developer.download.nvidia.com/compute/cuda/11.0.1/local_installers/cuda_11.0.1_450.36.06_linux.run
> sh cuda_11.0.1_450.36.06_linux.run
> vim ~/.bashrc
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/lib64
> export PATH=$PATH:/usr/local/cuda-11.0/bin
> export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.0
> source ~/.bashrc
> make CUDA_PATH=/usr/local/cuda-11.0 DEBUG=1
> python3 launch-backend.py ../bin/gem-schd ../bin/gem-pmgr --ip 127.0.0.1 --port 50051 --config ../resource-config.txt
> python3 launch-command.py --name client1 --port 50052 --ip 127.0.0.1 "/root/test"
> ```
>
> 目前实际拦截的 cuda API 有：
>
> + `cuLaunchCooperativeKernel`
> + `cuLaunchKernel`
> + `cuMipmappedArrayDestroy`
> + `cuMipmappedArrayCreate`
> + `cuArrayDestroy`
> + `cuArray3DCreate_v2`
> + `cuArrayCreate_v2`
> + `cuMemcpyDtoHAsync_v2`
> + `cuMemcpyDtoH_v2`
> + `cuMemcpyHtoD_v2`
> + 等等
>
> 要测试的是：
>
> 1. 注释掉 cudaEventCreate：
>
>    可以跑起来了！
>
>    客户端：
>
>    ![Gemini-image25]({{ site.url }}/my_img/Gemini-image25.png)
>
>    host 端：
>
>    ![Gemini-image26]({{ site.url }}/my_img/Gemini-image26.png)
>
> 2. 检查是否能够正确 prehook 和 posthook：在处理代码中增加调试输出信息，一旦真的进入这些函数，就会有相应的输出：
>
>    ![Gemini-image27]({{ site.url }}/my_img/Gemini-image27.png)
>
>    可以进入到 prehook 和 posthook 代码
>
> 3. **未解决**
>
>    另外，注意到输出的调试信息中，有一个 ERROR：
>
>    ![Gemini-image28]({{ site.url }}/my_img/Gemini-image28.png)
>
>    是在代码：
>
>    ```c++
>    CUdevice device;
>    CUresult rc = cuCtxGetDevice(&device); 
>    ```
>
>    返回值 rc 为 201，代表 `CUDA_ERROR_INVALID_CONTEXT`。这意味着尝试在无效或未初始化的 CUDA 上下文中执行操作。具体来说，`cuCtxGetDevice(&device)` 调用返回此错误，表明当前没有有效的 CUDA 上下文可供使用。
>
>    单独拿出来运行也会：
>
>    ![Gemini-image29]({{ site.url }}/my_img/Gemini-image29.png)
>
>    似乎是因为没有显式的 CUDA 上下文，在测试代码中，先创建上下文，然后运行 `cuCtxGetDevice(&device)` 就不会报错
>
>    尝试修改 hook.cpp，在初始化时创建上下文。
>
>    但是 cuCtxCreate 也会被拦截。运行起来卡在了 cuCtxCreate，所以是不是死锁了。
>
>    注释掉，只保留 cuInit，但还是有 `E/ failed to get current device: 201`
>
> 4. **TODO**
>
>    同时在两个 client 运行有大量计算需求的 cuda 代码，看是否真的限制住了。
>
>    前提是学习 profiling 工具

