FROM nvidia/cuda:8.0-cudnn6-devel
WORKDIR /root

COPY keyboard /etc/default/keyboard

# Get required packages
RUN apt-get update && \
  apt-get install vim \
                  python-pip \
                  python-dev \
                  python-opencv \
                  python-tk \
                  libjpeg-dev \
                  libfreetype6 \
                  libfreetype6-dev \
                  zlib1g-dev \
                  cmake \
                  wget \
                  cython \
                  git \
                  -y

# Get required python modules
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Update numpy
RUN pip install -U numpy

# Install python interface for COCO
RUN git clone https://github.com/pdollar/coco.git
WORKDIR /root/coco/PythonAPI
RUN make

RUN apt-get update
RUN apt-get install cuda nvidia-cuda-toolkit -y
RUN apt-get install libgeos-dev -y

WORKDIR /ai/tf-faster-rcnn

ADD ./lib  ./lib
ADD tools ./tools
ADD experiments ./experiments
ADD output ./output

# Add CUDA to the path
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV CUDA_HOME /usr/local/cuda
ENV PYTHONPATH /root/coco/PythonAPI
RUN ldconfig

WORKDIR /ai/tf-faster-rcnn/lib
RUN make clean
RUN make

WORKDIR /ai/tf-faster-rcnn