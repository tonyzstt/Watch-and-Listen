# Dataset Structure

## Overview
Dataset structures: training set as an example.

```
train/
├── audios
    ├── omK9uH2gfwg.wav
    ├── _gMR55WZfdo.wav
    ├── ...
├── images
    ├── omK9uH2gfwg
        ├── 0000.jpg
        ├── 0001.jpg
        ├── ...
    ├── _gMR55WZfdo
        ├── 0000.jpg
        ├── 0001.jpg
        ├── ...
    ├── ...
├── metas
    ├── omK9uH2gfwg.json
    ├── _gMR55WZfdo.json
```
* The `audios` folder contains the audio files of selected video segments.
* The `images` contains the image sequences of selected video segments.
* The `metas` folder contains metadata files of each video segment.
* please use video IDs in `train/test/val.json` to access the dataset.
* Run `bash run.sh` to get dataset.
* Access dataset [here](https://drive.google.com/file/d/10xPJOE83f7_18aJ9Jb9_SdPnvCIvrtbS/view?usp=sharing)

