# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.0-base
WORKDIR /artzucker/multilingual-pr
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .