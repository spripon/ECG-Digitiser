# Use an appropriate CUDA base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set the environment variable to avoid interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links to ensure 'python' and 'python3' point to 'python3.11'
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install the required dependencies for OpenCV
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install -y libgl1-mesa-glx
# RUN apt-get update && apt-get install -y \
#     git \
#     git-lfs \
#     libgl1-mesa-glx

# ## Initialize Git LFS
# RUN git lfs install --skip-repo
# RUN git lfs pull

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt

# Ensure setuptools is up-to-date
RUN pip install --upgrade pip setuptools

## Install nnUNet
WORKDIR nnUNet
RUN pip install .

## Move back out of nnUNet
WORKDIR ..

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda-12.1/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}

# Reset the DEBIAN_FRONTEND variable to default
ENV DEBIAN_FRONTEND=dialog