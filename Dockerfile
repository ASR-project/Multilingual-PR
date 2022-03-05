# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
WORKDIR /artzucker/multilingual-pr
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .