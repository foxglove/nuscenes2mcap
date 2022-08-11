# nuscenes2mcap

> _Convert [nuScenes](https://www.nuscenes.org/) data into [MCAP](https://mcap.dev/) format_

## Introduction

nuScenes is a large-scale dataset of autonomous driving in urban environments, provided free for non-commercial use. This project provides helper scripts to download the nuScenes dataset and convert scenes into ROS bag files for easy viewing in tools such as [Foxglove Studio](https://foxglove.dev/).

## Usage
1. Download one of the [nuScenes datasets](https://nuscenes.org/nuscenes). You will need to make
   an account and sign the terms of use.
1. Extract the following files into the `data/` directory:
    1. `can_bus.zip` to `data/`
    1. `nuScenes-map-expansion-v1.3.zip` to `data/maps`
    1. `v1.0-mini.tgz` to `data/`
1. Build and run the converter container with `./convert_mini_scenes.sh`

## License

nuscenes2mcap is licensed under [MIT License](https://opensource.org/licenses/MIT).

## Stay in touch

Join our [Slack channel](https://foxglove.dev/join-slack) to ask questions, share feedback, and stay up to date on what our team is working on.
