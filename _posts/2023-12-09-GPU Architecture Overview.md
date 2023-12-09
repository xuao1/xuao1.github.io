---
title: GPU Architecture Overview
author: xuao
date: 2023-12-09 16:12:00 +800
categories: [GPU]
tags: [GPU, GPU Architecture, GPU Contest, GPU Channel]
---

# GPU Architecture Overview

![gpu_management_model]({{ site.url }}/my_img/gpu_management_model.png)

主要参考 [GPU Architecture Overview](https://insujang.github.io/2017-04-27/gpu-architecture-overview/)

### 1 MMIO

CPU 通过 MMIO 与 GPU 进行通信。

DMA 支持传输大量数据，但是 **commands 是通过 MMIO**

### 2 GPU Context

context 代表了 GPU 计算的状态，拥有 GPU 中的虚拟地址空间。

GPU 上可以同时存在多个 active contexts

### 3 GPU Channel

任何 operation**（如 launch kernel）**都是通过 CPU 发出的 command 驱动的，

**command stream 被提交到 GPU channel 上**。每个 GPU context 可以拥有一个或多个 GPU channel，每个 GPU context 包含 GPU channel 描述符。

每个 GPU channel 描述符存储了 channel 的设置，包含一个 page table.

**每个 GPU channel 都有一个专用的 command buffer，该 buffer 分配在 GPU 内存中，通过 MMIO 对 CPU 可见。**

### 4 GPU Page Table

CPU contest 是通过 GPU page table 分配的，它将该 GPU context 的虚拟地址空间与其他的分隔开。

存储在 GPU 内存中，它的物理地址在 GPU channel 描述符中。

通过 channel 提交的所有 command 和 program 都在对应的 GPU 虚拟地址空间中执行。

GPU page table 将 GPU 虚拟地址转换为 GPU 物理地址以及主机物理地址，这使得能够将 GPU 内存和主机内存统一到 **unified GPU virtual address space**

### 5 PCIe BAR

PCIe 的基址寄存器，在 GPU boot 时配置，作为 MMIO 的窗口

GPU 控制寄存器和 GPU 内存映射到 BARs

Device driver 使用这些 MMIO 窗口来配置 GPU 并访问 GPU 内存。

### 6 PFIFO Engine

PFIFO 是 GPU command 提交时通过的特殊 engine，**维护多个独立的 command queues（即 channel）**。

command queue 是一个 ring buffer，有 put 和 get 指针.

对 channel 的所有访问都被 PFIFO 拦截并执行。

### 7 Bo (Buffer object)

是一块内存，可以存储纹理（texture）、渲染目标（render target）、着色器（shader）代码等等。