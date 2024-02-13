#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
FROM pytorch/pytorch:latest
#FROM anibali/pytorch:latest
#FROM nvcr.io/nvidia/pytorch:23.09-py3

WORKDIR /app

# install for mujoco support
RUN apt-get -y update
RUN apt-get install -y software-properties-common
RUN apt-get install -y libglfw3-dev
RUN apt-get install -y g++
#RUN sudo apt-get install -y libsdl2-dev libfreetype6-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev
#RUN sudo apt-get install -y python3-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libsdl2-dev libsmpeg-dev python3-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev
#RUN sudo apt-get install -y libjpeg-dev
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
