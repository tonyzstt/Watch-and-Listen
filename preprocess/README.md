# Preprocessing video and audio captions into conversational training data

## Prepare data

Change the two constants `MMTRAIL_DIR` and `QUESTIONS_DIR` in `preprocess.py` to your own directory.

```python
###
# TODO:
# Change the following two paths to your own paths 
# (where you store the MMTrail data and the alignment questions)
# and run `python preprocess.py`
# to preprocess the MMTrail metadata files.
# 
# The processed metadata will be stored under the same directory
# as the original metadata.
###
MMTRAIL_DIR = "$YOUR_MMTRAIL_DATA_PATH$"
QUESTIONS_DIR = "$YOUR_GIT_PROJECT_PATH$/preprocess"
```

The output of `tree -L 2` bash command under `MMTRAIL_DIR` directory before runing `preprocess.py` shall look like this:

```bash
.
├── test
│   ├── audios
│   ├── images
│   └── metas
├── test.json
├── train
│   ├── audios
│   ├── images
│   └── metas
├── train.json
├── val
│   ├── audios
│   ├── images
│   └── metas
└── val.json
```

The output of `tree -L 2` bash command under `MMTRAIL_DIR` directory after runing `preprocess.py` shall look like this:

```bash
.
├── test
│   ├── audios
│   ├── images
│   ├── metas
│   ├── metas_audio_convs.json
│   └── metas_video_convs.json
├── test.json
├── train
│   ├── audios
│   ├── images
│   ├── metas
│   ├── metas_audio_convs.json
│   └── metas_video_convs.json
├── train.json
├── val
│   ├── audios
│   ├── images
│   ├── metas
│   ├── metas_audio_convs.json
│   └── metas_video_convs.json
└── val.json
```

The output of `tree` bash command under `QUESTIONS_DIR` shall look like this:
```bash
.
├── README.md
├── alignment_questions_audio.txt
├── alignment_questions_video.txt
└── preprocess.py
```

## Run

After preparing your data and changing the constants in `preprocess.py` into your own path, run the following commands:

```bash
cd preprocess
python preprocess.py
```

