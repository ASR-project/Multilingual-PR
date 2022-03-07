# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:22.02-py3
WORKDIR /artzucker/multilingual-pr
COPY requirements_cuda11-3.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .