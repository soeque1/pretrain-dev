FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Install stand basic dependencies
RUN apt-get update && \
    apt-get -y install --only-upgrade bash && \
    apt-get -y install software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get -y install locales && locale-gen en_US.UTF-8 && locale-gen ko_KR.UTF-8 && \
    apt-get -y install git && \
    git config --global user.email "soeque1@gamil.com" && \
    git config --global user.name "soeque1" && \
    apt-get -y install wget && \
    apt-get -y install tar && \
    apt-get -y install gcc-6 g++-6 && \
    apt-get -y install build-essential libssl-dev zlib1g-dev libncurses5-dev libreadline-dev libgdbm-dev libdb5.3-dev libbz2-dev liblzma-dev libsqlite3-dev libffi-dev tcl-dev tk tk-dev && \
    wget https://www.python.org/ftp/python/3.7.7/Python-3.7.7.tar.xz && \
    tar xf Python-3.7.7.tar.xz && \
    cd Python-3.7.7 && \
    ./configure && \
    make altinstall

RUN ln -s /usr/local/bin/python3.7 /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py

ENV LANG='en_US.UTF-8' LANGUAGE='en_US.UTF-8' LC_ALL='en_US.UTF-8'

COPY requirements.txt /base
COPY prepare.sh /base

WORKDIR /base
RUN echo $PATH

RUN ./prepare.sh

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ['python', '-m', 'pytest', '-vv', 'tests/']
