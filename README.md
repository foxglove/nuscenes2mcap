# nuscenes2bag

> _Convert [nuScenes](https://www.nuscenes.org/) data into ROS [bag](http://wiki.ros.org/rosbag) format_

## Introduction

nuScenes is a large-scale dataset of autonomous driving in urban environments, provided free for non-commercial use. This project provides helper scripts to download the nuScenes dataset and convert scenes into ROS bag files for easy viewing in tools such as [Foxglove Studio](https://foxglove.dev/).

## Usage

    # This builds the `nuscenes2bag` Docker image and downloads various nuScenes datasets into `data/`
    $ ./setup.sh

    # Start a Jupyter notebook server on port 8888
    $ ./start.sh

    # Look in the console for the auth token and open a browser to http://localhost:8888/?token=<token>

## License

nuscenes2bag is licensed under [MIT License](https://opensource.org/licenses/MIT).

## Stay in touch

Join our [Slack channel](https://foxglove.dev/join-slack) to ask questions, share feedback, and stay up to date on what our team is working on.
