---
title: Paper Reading KubeShare
author: xuao
date: 2023-11-15 11:06:00 +800
categories: [Paper Reading]
tags: [GPU,GPU Sharing,Container,Kubernetes,GPU virtualization]
---

# KubeShare: A Framework to Manage GPUs as First-Class and Shared Resources in Container Cloud

## * 精简概括

### *.1 What

KubeShare，一个用于在 Kubernetes 中支持 GPU 共享的框架.

### *.2 Why

GPU 虚拟化在容器中却做得很有限。实现 GPU 共享以提高 GPU 资源利用率。

现有的支持在 Kubernetes 中进行 GPU 共享的解决方案，没有将 GPU 视为一流资源或提供局部性感知调度以减轻性能干扰问题。

### *.3 How

将 GPU 管理为一流资源，并提供了一个资源规范，允许用户使用 GPU 标识符请求其容器的特定 vGPU 绑定，并通过指定局部性约束来控制其 GPU 的位置.

位置约束用于控制 GPU 和容器之间的映射关系。支持三种类型的约束：**排除（exclusion）、亲和性（affinity）和反亲和性（anti-affinity）**.

通过基于 CUDA API 拦截的设备库实现，KubeShare 不仅可以隔离容器的 GPU 使用，还可以弹性地分配剩余的 GPU 容量给容器，而不会超过其最大需求.

使用一种**令牌（token）**的方式以**时间共享**的方式在容器之间隔离 GPU 的使用。令牌与时间配额关联。

后端模块是在主机上运行的独立守护程序，用于管理容器之间的令牌。

后端模块有三项主要任务：(1)跟踪每个容器的 GPU 使用时间；(2)将令牌分配给请求的容器之一；(3)确定令牌的时间配额。

## 0 Abstract

GPU 虚拟化在容器中却做得很有限.

其中一个关键挑战是缺乏支持多个并发容器之间共享 GPU 的机制，导致了 GPU 资源利用率较低

KubeShare，扩展了 Kubernetes 以实现 **GPU 的细粒度共享分配**，可以将GPU设备作为调度和分配的一流（first-class）资源

> 关键词：Cloud computing, GPU, Container, Scheduling

## 1 Introduction

虽然 Kubernetes 具备支持容器管理的强大功能，但 Kubernetes 本身只能原生识别和分配 **CPU** 和**内存**等计算资源.

要将任何其他自定义设备（包括 GPU、高性能网卡、FPGA 等）连接到容器，必须开发和安装**设备插件**。但设备插件不允许在自定义设备上进行资源共享或分配。

KubeShare，它扩展了 Kubernetes，以支持 GPU 的细粒度共享分配和一流资源管理。一流资源意味着资源实体可以被资源管理器和用户**明确识别和选择**。

KubeShare 允许用户在其分配的 GPU 上指定位置约束，以便 GPU 在容器之间共享时减少资源争用。

## 2 Background on Kubernetes

### 2.1 Architecture

在 Kubernetes 中，**可以创建和管理的最小可部署计算单元称为「Pod」**。

一个 Pod 代表一个逻辑主机，其中包含一个或多个容器，这些容器总是在同一位置部署、共享调度，并在共享上下文中运行。

论文假设每个 Pod 中只有一个容器。

Kubernetes 提供了一个模块化的 API 核心，允许用户以 YAML 或 JSON 对象的方式描述 Pod 的期望行为规范，这也被称为 PodSpec。

![Kubernentes architecture]({{ site.url }}/my_img/Kubernentes_arch.png)

主要组件：kube-apiserver，etcd，kube-scheduler，controller manager & controllers

节点组件：kubelet，container runtime，kube-proxy

### 2.2 Custom Device

Kubernetes 提供了一个设备插件框架，允许 kubelet 将其他自定义设备附加到 Pod 上.

## 3 Problems & Requirements

### 3.1 资源碎片化 Resource Fragmentation

设备插件框架是基于一个假设设计的，即自定义设备不能被分数分配或超额分配。因此，PodSpec 中自定义设备的资源需求必须是整数值.

一个欺骗 Kubernetes 框架的简单解决方案是将资源单位乘以一个缩放因子.

上述策略仍然可能受到资源碎片化问题的影响，因为 kube-scheduler 不将 GPU 视为一流资源，它只具有来自节点的聚合资源容量的信息，而没有关于个别设备的信息.

在共享环境下，如果不将 GPU 视为一流资源，使其具有唯一的标识符和使用状态，那么 GPU 资源将无法得到适当分配.

### 3.2 隐式和延迟绑定 Implicit and Late Binding

自定义设备与 Pod 之间的绑定既是隐式的，又是延迟的（即，在做出调度决策之后才确定）

kube-scheduler 无法控制将节点上的哪个设备分配给一个 Pod，而绑定决策直到 kubelet 在节点上创建 Pod 之后才会被确定

需要一种新的架构设计来支持 GPU 上的显式绑定和一流资源管理。控制自定义设备在 Pod 之间的共享方式，以实现可预测的性能和管理并发使用时的干扰。

### 3.3 Requirements

细粒度分配，资源隔离，位置感知，低开销，平台兼容性。

> 位置感知：
>
> 应该赋予用户控制 GPU 和容器之间绑定的能力，以减轻共享 GPU 上的性能干扰

## 4 Kubeshare

> KubeShare 在 GPU 共享、隔离、调度和兼容性方面都具有最强的支持

### 4.1 架构概述 Architecture Overview

KubeShare 的角色是**创建和管理 sharePod**，这是在 Kubernetes 中创建的一种**自定义资源类型**，用于表示具有附加共享自定义设备能力的 Pod。

为了实现 GPU 共享，KubeShare 从 Kubernetes 中分配物理 GPU，然后将它们分配给 sharePods，以实现分数分配。由 KubeShare 管理的这些共享 GPU 称为 vGPU。

+ Client 是希望将分数 GPU 资源分配给其容器的 Kubernetes 用户。客户端通过 kube-apiserver API 提交其资源规格与 KubeShare 进行交互
+ KubeShare-Sched 是根据当前资源状态和客户端指定的资源需求**决定容器和 vGPU 之间的映射关系的调度程序**，生成具有由调度策略决定的 GPUID 值的 SharePodSpec
+ KubeShare-DevMgr 创建 sharePod 对象，然后在接收到来自 KubeShare-Sched 的 SharePodSpec 后，在容器中初始化设备环境
+ vGPU 设备库：它是安装在每个容器中的库，通过拦截 CUDA 库中的所有与内存相关的 API 和计算相关的 API 来限制和隔离 GPU 的使用

### 4.2 First-class Resource Specification

KubeShare 将 GPU 视为一流资源，因此允许用户在规范中指定 GPU 的调度要求和约束

GPU 内存以空间共享，GPU 计算容量以时间片共享.

KubeShare 也支持 GPU 上的弹性资源分配。这意味着 KubeShare 将保证容器按照 gpu_request 指定的最小资源分配，并允许容器利用 GPU 上的剩余容量，只要其使用不超过 gpu_limit 的值.

位置约束用于控制 GPU 和容器之间的映射关系。支持三种类型的约束：**排除（exclusion）、亲和性（affinity）和反亲和性（anti-affinity）**.

+ 排除用于防止不同标签的容器之间共享 GPU
+ 亲和性强制要具有相同标签的容器被调度到同一个 GPU 上
+ 反亲和性是亲和性的相反约束，它强制要具有相同标签的容器被调度到不同的 GPU 上

### 4.3 资源感知和位置感知调度 Resource & Locality Aware Scheduling

依靠“**亲和性标签**”进行分配。具体实现是一个算法。

### 4.4 vGPU 生命周期管理 vGPU Lifecycle Manager

vGPU 的生命周期由 KubeShare-DevMgr 控制器管理，包括四个阶段：创建、活动、空闲和删除。

要将 vGPU 附加到容器中，KubeShare-DevMgr 扮演了与原生 Kubernetes 设备插件类似的角色。但是 KubeShare-DevMgr 根据 SharePodSpec 中的 GPUID 在容器和设备之间执行显式绑定。

### 4.5 GPU 使用隔离和弹性分配 GPU Usage Isolation & Elastic Allocation

使用一种**令牌（token）**的方式以**时间共享**的方式在容器之间隔离 GPU 的使用。令牌与时间配额关联。

解决方案由每个容器前端模块和每个节点后端模块组成。

前端模块是容器内的一个动态链接库。通过 **Linux LD_PRELOAD** 机制，它截取了与内存相关的所有 CUDA 库 API 和计算相关的 API（例如 cuLaunchKernel、cuLaunchGrid），这会强制应用程序在标准 GPU CUDA 库之前加载我们的设备库。

如果容器没有有效的令牌，前端模块将阻塞截取的 CUDA 调用，直到它从后端模块重新获取有效的令牌。

后端模块是在主机上运行的独立守护程序，用于管理容器之间的令牌。

在主机上只需要一个后端模块来独立管理每个设备的令牌。

后端模块有三项主要任务：(1)跟踪每个容器的 GPU 使用时间；(2)将令牌分配给请求的容器之一；(3)确定令牌的时间配额。

### 4.6 系统兼容性和灵活性 System Compatibility & Flexibility

我们的实现遵循 Kubernetes 定义的**运算符模式**。

运算符是 Kubernetes 定义的一种软件扩展框架，用于为特定类型的应用程序和资源对象提供自定义管理，而无需修改代码或更改现有 Kubernetes 集群的行为.

**运算符是自定义资源和自定义控制器的组合.**

**sharePod 是我们添加到 kube-apiserver 的自定义资源定义，而 KubeShare-DevMgr 是我们实现的控制器，用于创建和管理 sharePod.**

KubeShare 可以与其他 GPU 管理控制器或调度程序共存，并且对 vGPU 池外部的 GPU 没有影响.

KubeShare 将调度实现（即 KubeShare-Sched）和 GPU 共享实现（即 KubeShare-DevMgr）分解为两个独立的控制器。用户可以实现自己的调度逻辑或算法。

KubeShare 使 vGPU 成为 Kubernetes 中的一类一流资源，其中 vGPU 具有唯一的标识（即 GPUID）并且可以由用户明确请求。由于这个原因，KubeShare 能够支持资源描述中的局部性约束并启用局部性感知调度。

## 5 Experimental Evaluation

## 6 Related Works

论文提到了 3 种方案支持在 Kubernetes 中进行 GPU 共享，都采用了一种类似的方法来支持通过将 GPU 的资源单元乘以一个缩放因子来进行分配。

Deepomatic 仅支持分数分配，而未解决碎片化或资源隔离问题。

Aliyun 通过在 Kubernetes 中开发调度程序扩展程序来解决碎片化问题。但它只限制容器的 GPU 内存使用，而不限制计算使用。

GigaGPU 进一步扩展了 Aliyun，支持了基于 LD_PRELOAD API 拦截技术的内核执行上的 GPU 使用隔离。

## 7 Conclusions

KubeShare，一个用于在 Kubernetes 中支持 GPU 共享的框架.

将 GPU 管理为一流资源，并提供了一个简单但功能强大的资源规范，允许用户使用 GPU 标识符请求其容器的特定 vGPU 绑定，并通过指定局部性约束来控制其 GPU 的位置.

通过基于 CUDA API 拦截的设备库实现，KubeShare 不仅可以隔离容器的 GPU 使用，还可以弹性地分配剩余的 GPU 容量给容器，而不会超过其最大需求.

## # 概念学习

### #.1 DevOps

DevOps（Development 和 Operations 的缩写）是一种软件开发和 IT 运维（Operations）之间协同工作的文化和方法论。它旨在通过改进开发团队和运维团队之间的协作和沟通，以及采用自动化和持续集成/持续交付（CI/CD）等最佳实践，来加速软件开发、测试、部署和交付过程，以实现更高的效率、质量和可靠性。
