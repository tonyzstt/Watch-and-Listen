#!/bin/bash

# source ~/anaconda3/etc/profile.d/conda.sh
# conda init
# conda activate cs229-data

echo "Getting raw videos..."
python get_rawdata.py

echo "Extracting information..."
python process_video.py