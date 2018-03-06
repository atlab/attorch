FROM eywalker/pytorch-jupyter:cuda9.1

# Install latest DataJoint
RUN apt-get -y update && apt-get -y install ffmpeg libhdf5-10
RUN pip3 install imageio ffmpy h5py opencv-python
RUN pip3 install --upgrade git+https://github.com/datajoint/datajoint-python.git

ADD . /src/attorch
RUN pip3 install -e /src/attorch
