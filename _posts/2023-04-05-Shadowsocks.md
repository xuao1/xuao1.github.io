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

### 3. (可选)为服务器开启 BBR 加速

```shell
wget --no-check-certificate https://github.com/teddysun/across/raw/master/bbr.sh && chmod +x bbr.sh && ./bbr.sh
```

[一键为VPS开启BBR拥塞控制算法加速你的VPS网络速度](https://blog.csdn.net/weixin_39075913/article/details/129773890#:~:text=连接到你的VPS后，直接执行如下脚本一键开启BBR加速： wget --no-check-certificate https%3A%2F%2Fgithub.com%2Fteddysun%2Facross%2Fraw%2Fmaster%2Fbbr.sh %26%26,chmod %2Bx bbr.sh %26%26.%2Fbbr.sh 由于BBR加速只支持Linux内核版本4.9以上的，因此脚本会先升级系统内核，之后再开启BBR。)

### 3. 在需要使用 VPN 的主机上安装客户端

**下载对应客户端**

Windows：https://github.com/shadowsocks/shadowsocks-windows/releases

Mac：https://github.com/yangfeicheung/Shadowsocks-X/releases

Android：https://github.com/shadowsocks/shadowsocks-android/releases

