#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate cs229

echo "Getting raw videos..."
python get_rawdata.py

echo "Extracting information..."
python process_video.py