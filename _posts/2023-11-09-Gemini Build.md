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
> ![image-20231108213950485](C:\Users\15370\AppData\Roaming\Typora\typora-user-images\image-20231108213950485.png)
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

