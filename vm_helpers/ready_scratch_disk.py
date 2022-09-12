#!/usr/bin/env python3
import argparse
import json
import os
import sys
import subprocess


def fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def get_device_state(serial):
    lsblk_result = subprocess.run(["lsblk", "-Jo", "FSTYPE,SERIAL,MOUNTPOINT,PATH"], capture_output=True, check=True)
    block_devices = json.loads(lsblk_result.stdout)
    for device in block_devices["blockdevices"]:
        if device["serial"] == serial:
            return device
    fail(f"block device with serial '{serial}' not found in: {block_devices}")


def check_root():
    euid = os.geteuid()
    if euid != 0:
        fail(f"this script should be run as root, found euid {euid}")


def run(args, commit=False):
    if commit:
        subprocess.run(args, check=True)
    else:
        print(f"Would have executed: {args}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true", help="make destructive changes")
    parser.add_argument("--serial", default="nuscenes-scratch", help="device serial to use as scratch disk")
    parser.add_argument("--mountpoint", default="/media/scratch", help="mountpoint for scratch disk")
    args = parser.parse_args()
    if args.commit:
        check_root()
    state = get_device_state(args.serial)
    if state["fstype"] is None:
        run(["mkfs.ext4", "-F", state["path"]], commit=args.commit)

    state = get_device_state(args.serial)
    if state["mountpoint"] is None:
        if not os.path.isdir(args.mountpoint):
            if args.commit:
                os.makedirs(args.mountpoint, exist_ok=True, mode=0o777)
            else:
                print(f"Would have created directory {args.mountpoint}")
        run(["mount", state["path"], args.mountpoint], commit=args.commit)

    state = get_device_state(args.serial)
    if state["fstype"] != "ext4" and state["mountpoint"] != args.mountpoint:
        fail(f"could not completely set up device: {state}")


if __name__ == "__main__":
    main()
