FROM ros:noetic-ros-core

RUN apt-get update
RUN apt-get install -y git python3-pip python3-tf2-ros ros-noetic-foxglove-msgs libgl1 libgeos-dev
RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install shapely==1.8.* numpy==1.19 nuscenes-devkit mcap 'mcap-protobuf-support>=0.0.8' foxglove-data-platform tqdm requests protobuf
RUN pip3 install git+https://github.com/DanielPollithy/pypcd.git

COPY . /work

WORKDIR /work
