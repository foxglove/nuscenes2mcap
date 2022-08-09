FROM ros:noetic-ros-core

RUN apt-get update
RUN apt-get install -y git python3-pip python3-tf2-ros ros-noetic-foxglove-msgs libgl1
RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy==1.19 nuscenes-devkit mcap foxglove-data-platform tqdm requests
RUN pip3 install git+https://github.com/DanielPollithy/pypcd.git
