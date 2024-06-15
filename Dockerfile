FROM python:3.11

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libgl1-mesa-glx

## Initialize Git LFS
RUN git lfs install --skip-repo
RUN git lfs pull

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt

## Install nnUNet
WORKDIR nnUNet
RUN pip install -e .

## Move back out of nnUNet
WORKDIR ..