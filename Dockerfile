FROM eywalker/tensorflow-jupyter:v1.0.1-cuda8.0-cudnn5

RUN pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl && \
    pip install torchvision

RUN pip3 install jupyter

RUN pip3 install git+https://github.com/datajoint/datajoint-python.git

RUN apt-get update -y \
    && apt-get install -y graphviz \
    && pip3 install graphviz \
    && pip3 install gpustat

ADD . /src/attorch
RUN pip3 install -e /src/attorch

WORKDIR /notebooks


