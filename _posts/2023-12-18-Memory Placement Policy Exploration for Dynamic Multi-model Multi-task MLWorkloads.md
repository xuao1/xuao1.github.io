---
title: Paper Reading Memory Placement Policy Exploration for Dynamic Multi-model Multi-task ML Workloads
author: xuao
date: 2023-12-18 17:04:00 +800
categories: [Paper Reading]
tags: [Memory Placement,RT-MTMM,Mechine Learning,Heterogeneity]
---

# Memory Placement Policy Exploration for Dynamic Multi-model Multi-task ML Workloads

## * 精简概括

### *.1 What

针对 real-time multi-task multi-model (RT-MTMM) mechine learning 问题，通过建立模型，分析往 scratchpad 中放置模型的 weight parameters 的：

+ 放置策略 memoty placement strategy
+ 换出策略 eviction policy

对存储操作的能量 energy 消耗的影响

### *.2 Why

RT-MTMM 工作复杂有很多特性，包括 model heterogeneity, scenario-based consideration, periodic invocation, dynamic task graph. 这篇工作希望利用这些特性，测试不同的内存策略的效果。

### *.3 How

建模，一方面是 scratchpad 的建模，包括其大小，访存时间和能量消耗；另一方面是 RT MTMM workload 的建模，包括单一场景和组合场景。

放置策略包括：

+ 静态策略：选择最常用的 model layer，执行期间一直放在 scratchpad 中
+ 动态策略：根据统计信息中不同的指标，选择相应的 model layer 放入 scratchpad，配合 eviction policy

结果就是，动态放置策略更优，尤其是当 workload 的动态性增加时，但是 workload 的不同导致了动态策略最佳指标的选取不同。此外，scratchpad 的大小也有影响，较小的 scratchpad 不能明显体现出模型的异构性，所以不同策略之间差别不大。

### *.4 个人点评

这篇工作的核心是建模，分析不同策略。

比较的不是访存时间，而是访存 energy，因为 scratchpad 和 DRAM 之间，访存 energy 的差距比访存时间的差距大 5 倍，效果更明显。

动态策略测试时，与之对比的「静态策略 baseline」，不是之前提到的静态策略，而是：

> For the static baseline, we average over the possible model choices, accounting the unpredictable real-life deployment.

这样做效果更明显，尤其是 scratchpad 较小时。
