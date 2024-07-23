#!/bin/bash
# for x11 in docker.
DISPLAY="localhost:10.0"

# for nvidia
NVIDIA_VISIBLE_DEVICES=all  # 또는 특정 GPU ID
NVIDIA_DRIVER_CAPABILITIES=compute,utility  # 필요한 드라이버 기능
