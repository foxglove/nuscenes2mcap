# nuscenes2mcap

> _Convert [nuScenes](https://www.nuscenes.org/) data into [MCAP](https://mcap.dev/) format_

## Introduction

nuScenes is a large-scale dataset of autonomous driving in urban environments, provided free for non-commercial use. This project provides helper scripts to download the nuScenes dataset and convert scenes into ROS bag files for easy viewing in tools such as [Foxglove Studio](https://foxglove.dev/).

## Usage

* Create a VM in your GCS project.
    * Use a Debian base image
    * Give it the credentials of a service account with read access to the `gs://nuscenes_all`
      bucket. Ask James Smith for access.
    * :q
    :qa


* Download nuscenes data into the `data` folder and unpack:
    - can_bus.zip
    - v1.0-mini.tgz

## License

nuscenes2bag is licensed under [MIT License](https://opensource.org/licenses/MIT).

## Stay in touch

Join our [Slack channel](https://foxglove.dev/join-slack) to ask questions, share feedback, and stay up to date on what our team is working on.
