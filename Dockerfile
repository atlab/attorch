FROM eywalker/pytorch-jupyter

# Install latest DataJoint
RUN pip3 install --upgrade git+https://github.com/datajoint/datajoint-python.git

ADD . /src/attorch
RUN pip3 install -e /src/attorch
