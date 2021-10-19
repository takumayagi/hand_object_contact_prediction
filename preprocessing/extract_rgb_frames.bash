#! /bin/bash
#
# extract_frames.bash
# Copyright (C) 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.
#

VID=$1
PID=${VID:0:3}
VIDEO_DIR=data/videos/
OUT_DIR=data/rgb_frames/$PID/$VID

VIDEO_PATH=$VIDEO_DIR/$PID/${VID}.MP4
if [ ! -e $VIDEO_PATH ]
then
    echo "Video not found!"
    exit
fi

mkdir -p $OUT_DIR

if [ ${#VID} -eq 6 ]
then
    ffmpeg -i $VIDEO_PATH -vf "scale=-2:480" -q:v 4 -r 60 "$OUT_DIR/frame_%010d.jpg"
else
    ffmpeg -i $VIDEO_PATH -vf "scale=-2:480" -q:v 4 -r 50 "$OUT_DIR/frame_%010d.jpg"
fi
