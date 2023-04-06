---
title: Shadowsocks
author: xuao
date: 2023-04-05 23:17:00 +800
categories: [VPN]
tags: [Shadowsocks, VPN, Vultr]
---

简单记录下配置过程，利用 vultr 的服务器和 Shadowsocks 搭建 VPN

### 1. 在服务器上配置

1、执行如下命令

```shell
wget –no-check-certificate -O shadowsocks-all.sh https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks-all.sh
```

2、上面的命令执行结束后，执行下面的命令

```shell
chmod +x shadowsocks-all.sh
```

3、上面的命令执行结束后，执行下面的命令

```shell
./shadowsocks-all.sh 2>&1| tee shadowsocks-all.log
```

4、执行上述命令会有相关输入提示操作。根据需要选择。不明白的话就直接选 1 或者直接默认回车。之后会提示你输入密码和端口，对应设置即可，或者直接使用默认的。全部执行完成之后就会出现如下信息：

```bash
StartingShadowsocks success

Congratulations, Shadowsocks-Python server install completed!

YourServer IP : 你的IP

YourServerPort: 在第四步提示设置的端口号

YourPassword: 在第四步提示设置的密码

YourEncryptionMethod: aes-256-cfb

Your QR Code: (ForShadowsocksWindows, OSX, Androidand iOS clients)

ss://YWVzLTI1Ni1jZmI6emh1aTA4MTA0MTJaaaccuMjmmLjU1LjE5MTo4tdVg4

Your QR Code has been saved as a PNG file path:

/root/shadowsocks_python_qr.png

Welcome to visit: https://teddysun.com/486.html

Enjoy it!
```

5、看到以上信息就说明安装完成了，然后根据不同的终端设备进行设置就可以了

6、重新启动：

```shell
/etc/init.d/shadowsocks-python restart
```

### 2. 开放端口

开放对应的 TCP 端口

需要两个步骤：

1、在服务器管理页面添加防火墙规则， 打开该端口

2、在服务器内开启端口：

```shell
ufw enable
ufw allow 15117
ufw enable
```

或者：

```shell
apt install firewalld
firewall-cmd --zone=public --add-port=15117/tcp --permanent
firewall-cmd --reload
systemctl restart firewalld.service
```

3、可通过该网站测试端口是否开启：[Open Port Check Tool - Test Port Forwarding on Your Router](https://www.yougetsignal.com/tools/open-ports/)

### 3. (可选)配置加速

#### 3.1 为服务器开启 BBR 加速

```shell
wget --no-check-certificate https://github.com/teddysun/across/raw/master/bbr.sh && chmod +x bbr.sh && ./bbr.sh
```

[一键为VPS开启BBR拥塞控制算法加速你的VPS网络速度](https://blog.csdn.net/weixin_39075913/article/details/129773890#:~:text=连接到你的VPS后，直接执行如下脚本一键开启BBR加速： wget --no-check-certificate https%3A%2F%2Fgithub.com%2Fteddysun%2Facross%2Fraw%2Fmaster%2Fbbr.sh %26%26,chmod %2Bx bbr.sh %26%26.%2Fbbr.sh 由于BBR加速只支持Linux内核版本4.9以上的，因此脚本会先升级系统内核，之后再开启BBR。)

#### 3.2 增加系统文件描述符的最大限数

编辑文件 `limits.conf`

```shell
vi /etc/security/limits.conf
```

增加以下两行

```shell
* soft nofile 51200
* hard nofile 51200
```

启动shadowsocks服务器之前，设置以下参数

```shell
ulimit -n 51200
```

#### 3.3 调整内核参数

修改配置文件 `/etc/sysctl.conf`

```bash
fs.file-max = 51200

net.core.rmem_max = 67108864
net.core.wmem_max = 67108864
net.core.netdev_max_backlog = 250000
net.core.somaxconn = 4096

net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_tw_recycle = 0
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 1200
net.ipv4.ip_local_port_range = 10000 65000
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.tcp_max_tw_buckets = 5000
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_rmem = 4096 87380 67108864
net.ipv4.tcp_wmem = 4096 65536 67108864
net.ipv4.tcp_mtu_probing = 1
net.ipv4.tcp_congestion_control = hybla
```

修改后执行 `sysctl -p` 使配置生效

#### 3.4 使用魔改 BBR

魔改BBR是原版BBR基础上的第三方激进版本，效果优于原版BBR。

```shell
wget "https://github.com/cx9208/Linux-NetSpeed/raw/master/tcp.sh" 
chmod +x tcp.sh 
sudo ./tcp.sh
```

先选 2 执行安装，然后 7 开启

提示 remove 时，选 no

### 3. 在需要使用 VPN 的主机上安装客户端

**下载对应客户端**

Windows：https://github.com/shadowsocks/shadowsocks-windows/releases

Mac：https://github.com/yangfeicheung/Shadowsocks-X/releases

Android：https://github.com/shadowsocks/shadowsocks-android/releases

