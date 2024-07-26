### 获取在X86主机上进行交叉编译的Python3

#### 1. 安装下载工具dfss

```bash
pip3 install dfss --upgrade
```

#### 2. 根据版本使用dfss下载编译所需要的Python3

* Python3.5
    ```
    python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.5.9.tar.gz
    ```

* Python3.6
    ```
    python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.6.5.tar.gz
    ```

* Python3.7
    ```
    python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.7.3.tar.gz
    ```

* Python3.8
    ```
    python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.8.2.tar.gz
    ```

* Python3.9
    ```
    python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.9.0.tar.gz
    ```

* Python3.10
    ```
    python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.10.0.tar.gz
    ```

* Python3.11
    ```
    python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.11.0.tar.gz
    ```

