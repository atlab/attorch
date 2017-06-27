FROM eywalker/pytorch-jupyter

RUN pip install git+https://github.com/datajoint/datajoint-python.git

RUN apt-get update -y \
    && apt-get install -y graphviz \
    && pip install graphviz \
    && pip install gpustat

ADD . /src/attorch
RUN pip install -e /src/attorch

WORKDIR /notebooks


