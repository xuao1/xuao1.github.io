---
title: 图计算学习笔记
author: xuao
date: 2023-04-13 11:19:00 +800
categories: [Graph Computing]
tags: [Graph]
---

# 图计算

## 一、基本概念

### 1. 定义

“图计算”中的“图”指的是数据结构，针对“图论”而非图像。图 G 由节点 V（vertice）与边 E（edge）构成，一般表示为 G（V，E）。

图数据对一组对象（顶点）及其关系（边）进行建模，可以直观、自然地表示现实世界中各种实体对象以及它们之间的关系。

**图分析：**图分析使用基于图的方法来分析连接的数据。我们可以查询图数据，使用基本统计信息，可视化地探索图、展示图，或者将图信息预处理后合并到机器学习任务中。图的查询通常用于局部数据分析，而图计算通常涉及整张图和迭代分析。

狭义的图计算说的是，在确定不变的图上面来做各种各样的计算。

而广义的图计算能干的事会更多一些，它是指基于图数据来做各种各样的处理。比如说这张图是在变化的。在路网图上，如果只把路的宽度作为边的属性，那它可能就是很长时间不变的，但如果我们以边上路的拥堵程度作为属性的话，那这个图是在变的。还比说流图计算，数据源源不断地以流的方式到来，该怎么处理。这些都是属于广义的图计算范畴。

广义的图计算，可以理解是包括了图数据库。

目前大数据处理框架比如 MapReduce 和 Spark，他们的主要作用是大规模数据分析，以批处理计算为主，其实时性需求得不到满足。在大数据应用场景下，数据价值会随着时间的流逝而衰减，因此期望能够尽快堆最新的数据做出分析并给出结果，并实时展示，以达到实时计算。

### 2. 优点

+ 图数据可以很好地描述事物之间的联系，包括描述联系的方向和属性。
+ 图是认识世界的一种新的方式：
  + 从数据结构的演进上看，**图是对事物之间关系的一种原生的（native）表达**，它用来表示关联的时候，它的描述能力要比链表、树这些数据结构要强得多。**所以说当用图来认识世界，特别是用来认识关联的时候，它是一种更加直接和先进的方式。**
  + 传统的数据库叫关系数据库，但其实它的数据组织形式并不是关系原生的，而是以表的形式进行组织，就是一张一张的表，然后靠表上的键连起来的。所以其实关系数据库应该叫表数据库，而图数据库反而应该叫关系数据库。

### 3. 类别

#### 3.1 图的交互查询

在图计算的应用中，业务通常需要以探索的方式来查看图数据，以进行一些问题的及时定位和分析某个深入的信息

#### 3.2 图分析

关于图分析计算的研究已经持续了数十年，产生了很多图分析的算法。典型的图分析算法包括经典图算法（例如，PageRank、最短路径和最大流），社区检测算法（例如，最大团/clique、联通量计算、Louvain 和标签传播），图挖掘算法（例如，频繁集挖掘和图的模式匹配）。由于图分析算法的多样性和分布式计算的复杂性，分布式图分析算法往往需要遵循一定的编程模型。当前的编程模型有点中心模型“Think-like-vertex”，基于矩阵的模型和基于子图的模型等。在这些模型下，涌现出各种图分析系统，如 Apache Giraph、Pregel、PowerGraph、Spark GraphX、GRAPE 等。

#### 3.3 基于图的机器学习

经典的 Graph Embedding 技术，例如 Node2Vec 和 LINE，已在各种机器学习场景中广泛使用。近年来提出的图神经网络（GNN），更是将图中的结构和属性信息与深度学习中的特征相结合。GNN 可以为图中的任何图结构（例如，顶点，边或整个图）学习低维表征，并且生成的表征可以被许多下游图相关的机器学习任务进行分类、链路预测、聚类等。图学习技术已被证明在许多与图相关的任务上具有令人信服的性能。与传统的机器学习任务不同，图学习任务涉及图和神经网络的相关操作（见图 2 右），图中的每个顶点都使用与图相关的操作来选择其邻居，并将其邻居的特征与神经网络操作进行聚合。

### 4. 常用工具：

传统图计算工具大致分为以下类型

+ 图数据库：ArangoDB、Amazon Neptune、Neo4j、Orient DB、Dgraph、FlockDB
+ 图分析平台：TigerGraph、BigGraph
+ 图计算引擎：GraphX、Giraph

### 5. 图计算场景中常遇到的问题

- 图计算问题十分复杂，计算模式多样，解决方案碎片化。
- 图计算学习难度强，成本大，门槛高。
- 图的规模和数据量大，计算复杂，效率低。

### 6. 图计算框架

#### 6.1 单机内存图处理系统

此类图计算系统单机运行，可直接将图完全加载到内存中进行计算。

只能解决小规模的图计算问题。

+ Ligra：提出根据图形稠密情况自适应的切换计算模式，并提供了一种基于边映射，定点映射以及顶点集映射的并行编程算法。
+ Galois：使用 DSLs 写出更复杂的算法完成图分析工作，并发现当输入图是道路网络或者具有较大直径的图时能获得一个数量级的性能提升。
+ GraphMap：第一个对多核 CPU 进行由阿胡的以顶点为编程中心的轻量级图计算框架。
+ Polymer：针对在 NUMA 特性的计算机结果上运行图算法的优化。无论是随即或者交错分配图数据都会重大地束缚数据地本地性和并行性。无论是 intra-node 还是 inter-node，顺序访问都比随机访存地带宽高得多。

#### 6.2 单机核外图处理系统

单机运行，但是将存储层次由 RAM 扩展到外部存储器。

有 GraphChi, TurboGraph, X-Stream, PathGraph, GridGraph 和 FlashGraph。

#### 6.3 分布式内存图处理系统

将图数据全部加载到集群中的内存中计算。

图分割的挑战在分布式系统愈加明显，再加上集群网络总带宽的限制，所以整体性能和所能处理的图规模也存在一定的缺陷。

这类图计算系统主要包括同步计算模型的 Pregel 及其开源实现Piccolo，同时支持同步和异步的系统 PowerGraph，GraphLab 和 GraphX。PowerSwitch 和 PowerLyra 则对 PowerGraph 做了改进, Gemini 则借鉴了单机内存系统的特性提出了以计算为核心的图计算系统。

+ Pregel：采用 BSP 计算模型的分布式内存图计算系统，计算由一系列超步组成。
+ GraphLab：针对 MLDM 应用。
+ PowerGraph：针对 power-low 特性的自然图详细分析工作负载，图分割，通信，存储和计算等各方面带来的挑战。提供了完善的图分割数学理论支撑，证明切点法比切边法能提高一个数量级的图计算性能。故 PowerGraph 使用 p-way 切点法，采用了以顶点为中心的 GAS 编程模型，增加了细粒度并发性同时支持同步和异步模型。
+ PowerLyra：从图分割方面对 PowerGraph 进行了改进。提出了一种混合图分割方法 hybrid-cut，即出入度高的顶点采用切点法反之出入度低的顶点采用切边法，经过试验对比性能提高了至少 1.24 倍。
+ PowerSwitch：从同异步模型方面对 PowerGraph 进行了改进。提出了一种混合图计算模型 Hsync。
+ Gemini：以计算为中心的图计算系统。它针对图结构的稀疏或稠密情况使用于与 Ligra 相同的自适应 push/pull 方式的计算，并在现代 NUMA-aware 特性的内存中采用基于 chunk 的图划分进行更细粒度的负载均衡调节。

#### 6.4 分布式核外图处理系统

此类图计算系统将 Single-machine out-of-core systems 拓展为集群，能够处理边数量级为 trillion 的图。

### 7. 图关键技术

#### 7.1 图数据的组织

由于实际图的稀疏性，图计算系统通常使用稀疏矩阵的存储方法来表示图数据，其中最常用的两种是CSR（Compressed Sparse Row）和CSC（Compressed Sparse Column），分别按行（列）存储每行（列）非零元所在列（行），每一行则（列）对应了一个顶点的出边（入边）。

#### 7.2 图数据的划分

将一个大图划分为若干较小的子图，是很多图计算系统都会使用的扩展处理规模的方法；此外，图划分还能增强数据的局部性，从而降低访存的随机性，提升系统效率。

对于分布式图计算系统而言，图划分有两个目标：

1. 每个子图的规模尽可能相近，获得较为均衡的负载。
2. 不同子图之间的依赖（例如跨子图的边）尽可能少，降低机器间的通信开销。

图划分有按照顶点划分和按照边划分两种方式，它们各有优劣：

1. 顶点划分将每个顶点邻接的边都放在一台机器上，因此计算的局部性更好，但是可能由于度数的幂律分布导致负载不均衡。
2. 边划分能够最大程度地改善负载不均衡的问题，但是需要将每个顶点分成若干副本分布于不同机器上，因此会引入额外的同步/空间开销。

#### 7.3 顶点程序的调度

在以顶点为中心的图计算模型中，每个顶点程序可以并行地予以调度。大部分图计算系统采用基于BSP模型的同步调度方式，将计算过程分为若干超步（每个超步通常对应一轮迭代），每个超步内所有顶点程序独立并行地执行，结束后进行全局同步。顶点程序可能产生发送给其它顶点的消息，而通信过程通常与计算过程分离。

同步调度容易产生的问题是：

1. 一旦发生负载不均衡，那么最慢的计算单元会拖慢整体的进度。
2. 某些算法可能在同步调度模型下不收敛。

为此，部分图计算系统提供了异步调度的选项，让各个顶点程序的执行可以更自由，例如：每个顶点程序可以设定优先级，让优先级高的顶点程序能以更高的频率执行，从而更快地收敛。 然而，异步调度在系统设计上引入了更多的复杂度，例如数据一致性的维护，消息的聚合等等，很多情况下的效率并不理想。

#### 7.4 通信模式

主要分为两种，推动（Push）和拉取（Pull）：

1. 推动模式下每个顶点沿着边向邻居顶点传递消息，邻居顶点根据收到的消息更新自身的状态。所有的类 Pregel 系统采用的几乎都是这种计算和通信模式。
2. 拉取模式通常将顶点分为主副本和镜像副本，通信发生在每个顶点的两类副本之间而非每条边连接的两个顶点之间。GraphLab、PowerGraph、GraphX等采用的均为这种模式。

除了通信的模式有所区别，推动和拉取在计算上也有不同的权衡：

1. 推动模式可能产生数据竞争，需要使用锁或原子操作来保证状态的更新是正确的。
2. 拉取模式尽管没有竞争的问题，但是可能产生额外的数据访问。

## 二、现有框架调研

### 1. Wukong

> Wukong: A Distributed Framework for Fast and Concurrent Graph Querying

#### 1.1 是什么

这篇论文介绍了一个名为 Wukong 的分布式图查询框架，它旨在加速大规模图数据的查询和分析。Wukong 提供了一个灵活的图查询语言，支持基于节点、边和路径的复杂查询，同时还提供了并发查询和高效数据加载的支持。Wukong 采用了一种基于内存的数据存储方式和分布式计算模型，可以快速处理大型图数据。实验结果表明，Wukong 比目前流行的图查询系统在性能和可扩展性方面都表现更好。

**Wukong**: **a distributed in-memory framework** that provides low latency and high throughput for **concurrent query** processing over large and fast-evolving **graph data**.

#### 1.2 为什么

With the increasing scale of data volume and the growing number of concurrent requests, running queries over distributed in-memory graph store becomes essential but also challenging.

Prior work has prominent limitations of query performance in many scenarios.

+ 使用关系数据库处理图查询时，会依赖 join 操作，这会导致大量的冗余中间数据。
+ 无法及时处理
+ 查询的并行化不好，甚至会导致阻塞
+ It is difficult to partition the graph with good locality.

#### 1.3 怎么做

key techniques：RDMA networking and hardware heterogeneity.

**Wukong:** a distributed in-memory graph query system, which fully leverages(利用) one-sided RDMA  primitives to support fast and concurrent queries over billion-scale graphs.

**Wukong+S:** 采用一种综合设计，其中包括新的优化，支持在快速演化的图中进行连续和单次查询的混合查询。

**Wukong+G:** 利用 CPU/GPU 集群的异构 with RDMA-capable networks 来加速混合工作。

**Wukong+M: **采用了一种细粒度的实时迁移方案，以保持对动态图并发查询操作的局部性。

#### 1.4 概念补充

**Distributed in-memory graph**

分布式内存图，其中图数据存储在多个计算节点的内存中，并且在这些节点之间进行并行处理。

这种类型的图具有以下优点：

+ 内存存储可以提高图查询的速度，因为内存的访问速度比磁盘快得多。
+ 分布式计算模型可以加速图查询处理，因为多个计算节点可以并行处理不同的任务。
+ 分布式内存图还具有可扩展性和容错性，可以在需要处理更大的图数据或在节点故障时保持高性能。

### 2. Wukong+G 

> Fast and concurrent RDF queries using RDMA-assisted GPU graph exploration

#### 2.1 是什么

**The first graph-based distributed RDF query processing system that efficiently exploits the hybrid parallelism of CPU and GPU.**

**基于基于 RDMA 的高并发、快速的分布式 RDF Graph Query 系统。**

RDMA：Remote Direct Memory Access，即远程直接内存访问。它是一种零拷贝、低延迟、高带宽的网络传输技术，可用于在计算机系统之间高效地传输数据。

RDF：资源描述框架，将现实生活中的关系描述成实体与实体之间的关系，用图来表示。

通过对大量且不断增长的 RDF 数据进行大量查询，RDF 图形存储库为并发查询处理提供低延迟和高吞吐量势在必行。

#### 2.2 为什么

RDF 图规模的急剧增加对大型 RDF 数据集上的快速和并发查询提出了巨大的挑战；许多 RDF 查询并行性不好，查询的异构性可以导致现有 RDF 存储库的巨大延迟差异；GPU 被设计为提供大规模简单控制流操作的高计算吞吐量，具有很少或没有控制依赖性。

#### 2.3 怎么做

Wukong+G 通过三个关键设计使其快速且并发。

首先，Wukong+G 利用 GPU 来控制图形探索中的随机内存访问，通过有效地映射 CPU 和 GPU 之间的数据来实现延迟隐藏，包括一系列技术，如查询感知预取、模式感知流水线和细粒度交换。

其次，Wukong+G 通过引入 GPU 友好的 RDF 存储来扩展规模，以支持超过 GPU 内存大小的 RDF 图表，通过使用基于谓词的分组、成对缓存和前瞻替换等技术来缩小主机和设备内存规模之间的差距。

第三，Wukong+G 通过通信层扩展规模，该层将查询元数据和中间结果的传输过程解耦，并利用本地和 GPUDirect RDMA 在 CPU/GPU 集群上实现有效的通信。

### 3. GridGraph

#### 3.1 是什么

> Zhu X, Han W, Chen W. GridGraph: Large-Scale Graph Processing on a Single Machine Using 2-Level Hierarchical Partitioning[C]//USENIX Annual Technical Conference. 2015: 375-386.



### 4. Gemini

#### 4.1 是什么

> Zhu X, Chen W, Zheng W, et al. Gemini: A computation-centric distributed graph processing system[C]//12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), Savannah, GA. 2016.

针对已有系统的局限性，提出了**以计算为中心**的设计理念，通过降低分布式带来的开销并尽可能优化本地计算部分的实现，使得系统能够在具备扩展性的同时不失高效性。

#### 4.2 为什么

#### 4.3 怎么做

Gemini 采用了基于顶点划分的方法来避免引入过大的分布式开销；但是在计算模式上却借鉴了边划分的思想，将每个顶点的计算分布到多台机器上分别进行，并尽可能让每台机器上的计算量接近，从而消解顶点划分可能导致的负载不均衡问题。

### 5. Cyclops

#### 5.1 是什么

> Chen R, Ding X, Wang P, et al. Computation and communication efficient graph processing with distributed immutable view[C]//Proceedings of the 23rd international symposium on High-performance parallel and distributed computing. ACM, 2014: 215-226.



### 6. PowerLyra

#### 6.1 是什么

> Chen R, Shi J, Chen Y, et al. Powerlyra: Differentiated graph computation and partitioning on skewed graphs[C]//Proceedings of the Tenth European Conference on Computer Systems. ACM, 2015: 1.

现有的图数据，如社交网络、Web 网页等都是一种 Power-law 幂律图的特征。

所谓 Power-law 幂律图就是指在图数据中顶点的度数分配不均匀。有的图顶点的度数很高，有的顶点度数很低。并且顶点度数呈现着幂律分布的特征，对于这种 Power-law 的图数据，会存在很大的计算分配不均匀的特征。

PowerLyra 采用了一种 Hybrid 切分策略，对于高度顶点采用 vertex-cut 的策略，对于低度顶点采用 edge-cut 的切分方式。

大多数现有的图处理系统都遵循一个顶点哲学：设计一个图处理算法作为一个**以顶点为中心的图处理程序**去并行处理每个顶点，并且**沿着这些顶点的边进行通信**。

#### 6.2 为什么？

程序是在本地去执行的。所以一个最直接的方法就是使资源在本地可以访问的到所以去降低网络延迟。然而对于高度的顶点去聚集它的followers的资源并且当接受到来自邻居节点激活顶点消息，高度的顶点会成为高度抢占的中心。这样对于那些度很高的顶点会造成严重的负载不均衡和网络通信开销。这样唯一的解决方案就是将这样一个高负载的顶点分开到多个机器中去使其负载均衡。并且他们能够被并行执行。然而这样的分开策略将会带来更多的通信和计算。

低度顶点希望顶点计算 LOCALITY 更好，具体来说，就是低度顶点希望顶点计算都能够在本地执行，并且能够隐藏网络延迟。

高度顶点希望所有的顶点被并行的执行，高度顶点被切分成多个 Mirror 顶点来防止顶点执行负载不均衡。

在高度顶点 PARALLELISM 和低度顶点 LOCALITY 之间，存在一个 locality 和 Parallelism 的冲突选择。

#### 6.3 怎么做？

提供了一种混合切分策略：对于一些低度的顶点保持他们的 locality。对于一些高度的顶点保持他们的 parallelism。他既提高了 locality 并且减少了通信带来的开销。

### 7. PowerSwitch

#### 7.1 是什么

> Xie C, Chen R, Guan H, et al. Sync or async: Time to fuse for distributed graph-parallel computation[J]. ACM SIGPLAN Notices, 2015, 50(8): 194-204.



### 8. 腾讯 Plato

[Tencent/plato: 腾讯高性能分布式图计算框架Plato (github.com)](https://github.com/tencent/plato)

### 9. 阿里 GraphScope

#### 9.1 是什么

一站式图计算平台。

GraphScope 提供 Python 客户端，能十分方便的对接上下游工作流，具有一站式、开发便捷、性能极致等特点。它具有高效的跨引擎内存管理，在业界首次支持 Gremlin 分布式编译优化，同时支持算法的自动并行化和支持自动增量化处理动态图更新，提供了企业级场景的极致性能。

**底层：** 分布式内存数据管理系统 vineyard。在跨引擎之间，图数据按分区的形式存在于 vineyard，由 vineyard 统一管理。

**中间层：** 引擎层，分别由交互式查询引擎 GIE，图分析引擎 GAE，和图学习引擎 GLE 组成

**上层：**开发工具和算法库。GraphScope 提供了各类常用的分析算法，包括连通性计算类、社区发现类和 PageRank、中心度等数值计算类的算法。此外也提供了丰富的图学习算法包，内置支持 GraphSage、DeepWalk、LINE、Node2Vec 等算法。

### 10. 字节 ByteGraph



## 参考

[开源！一文了解阿里一站式图计算平台GraphScope-阿里云开发者社区 (aliyun.com)](https://developer.aliyun.com/article/780137?utm_content=g_1000241609)

[国内图计算研究哪里比较强](https://www.zhihu.com/question/49889826/answer/148332420)

[图计算，下一个科技前沿? ](https://zhuanlan.zhihu.com/p/528091760)

[字节跳动自研万亿级图数据库 & 图计算实践 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/109401046)

[大规模图计算系统综述 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/38010945)

[关于图计算&图学习的基础知识概览：前置知识点学习（PGL 系列一 - 飞桨AI Studio (baidu.com)](
