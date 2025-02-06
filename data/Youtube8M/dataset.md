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
├── videos
    ├── omK9uH2gfwg
        ├── metadata.json
        ├── video.mp4.mkv
    ├── _gMR55WZfdo
        ├── metadata.json
        ├── video.mp4
    ├── ...
├── extracted_sequences.json
```
* The `audios` folder contains the audio files of selected video segments.
* The `images` contains the image sequences of selected video segments.
* The `videos` folder contains raw (partial) videos from the dataset. Some may be invalid due to authorization or network issues.
* `extracted_sequences.json` contains a list of valid video IDs.
* Run `bash run.sh` to get dataset.
* Access dataset here: [training](https://drive.google.com/file/d/1WaxWSGJxufNLjWNySwkZJPttpDvZqInV/view?usp=sharing), [testing](https://drive.google.com/file/d/1z2GBRO7UD87ABRMB7yG2oT3ZgqIj_fPY/view?usp=sharing)

