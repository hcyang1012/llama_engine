FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Add user as sudoer
RUN useradd -m -s /bin/bash -G sudo user
RUN echo "user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers


RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y lsb-release

RUN apt-get install -y git python3 vim gcc g++ build-essential openssl libssl-dev wget gcovr lcov cmake lsb-release software-properties-common gnupg apt-utils python3-pip

RUN pip3 install accelerate sacrebleu scikit-learn evaluate transformers peft sqlitedict pytablewriter