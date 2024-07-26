.. _Get Python3 for cross-compilation on the X86 host:

Get Python3 for cross-compilation on an X86 host
======================================================================================

**1. Install the download tool dfss**

    .. parsed-literal::

        pip3 install dfss --upgrade

**2. Use dfss to download the Python3 required for compilation according to the version**

- Python3.5

    .. parsed-literal::

        python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.5.9.tar.gz

- Python3.6

    .. parsed-literal::

        python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.6.5.tar.gz


- Python3.7

    .. parsed-literal::

        python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.7.3.tar.gz


- Python3.8

    .. parsed-literal::

        python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.8.2.tar.gz


- Python3.9

    .. parsed-literal::

        python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.9.0.tar.gz

- Python3.10

    .. parsed-literal::

        python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.10.0.tar.gz

- Python3.11

    .. parsed-literal::

        python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.11.0.tar.gz






