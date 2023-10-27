---
title: Extend Vmware Disk
author: xuao
date: 2023-10-27 20:32:00 +800
categories: [Vmware]
tags: [VMware]
---

# VMware 虚拟机扩展硬盘大小

1. **增加 VMware 虚拟硬盘容量**：

   关闭 Ubuntu 虚拟机，在设置中更改硬盘大小，保存设置

2. **在 Ubuntu 内部扩展文件系统**：

   安装 gparted：

   ```shell
   sudo apt install gparted
   ```

   启动 gparted：

   ```shell
   sudo gparted
   ```

   在 gparted 中，选择要扩展的分区，然后点击“调整/移动”。

   调整分区大小

   点击“应用所有操作”按钮来应用更改。

3. 如果扩展的是 root 分区，可能还需要调整文件系统大小。使用以下命令扩展文件系统（以ext4为例）：

   ```
   sudo resize2fs /dev/sdXN
   ```

   其中，`/dev/sdXN` 是分区名称。例如，它可能是`/dev/sda3`。

4. 重启 Ubuntu 虚拟机