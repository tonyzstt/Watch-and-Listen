from typing import Tuple, Optional
import json
import os
import random


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
MMTRAIL_DIR = "/home/tonyzst/Desktop/CS229-Project/data/MMTrail"
QUESTIONS_DIR = "/home/tonyzst/Desktop/CS229-Project/preprocess"


def load_questions(root_dir: str) -> Tuple[list, list]:
    """
    Load random questions for video and audio captioning
    """
    video_questions = []
    with open(os.path.join(root_dir, 'alignment_questions_video.txt')) as f:
        for line in f:
            video_questions.append(line.strip())
    
    audio_questions = []
    with open(os.path.join(root_dir, 'alignment_questions_audio.txt')) as f:
        for line in f:
            audio_questions.append(line.strip())
    
    return video_questions, audio_questions


def preprocess_MMTrail_metadata(dataset_root_dir: str):
    """
    Preprocess MMTrail metadata files and store processed metadata in json format
    """
    
    # Directory structure:
    # your_dataset_root_dir
    # ├── train/val/test
    # │   ├── metas
    # │   │   ├── <id>.json
    # │   │   ├── ...
    # │   ├── metas_video_convs.json
    # │   ├── metas_audio_convs.jspn
    
    video_questions, audio_questions = load_questions(QUESTIONS_DIR)
    
    def preprocess(meta: dict) -> Optional[Tuple[dict]]:
        """
        Retrieve necessary information from metadata and construct conversation pairs
        """
        
        try:
            # process video metadata
            meta_processed_video = {
                'id': meta['video_id'],
                'video_path': meta['original_metadata']['video_path'],
                'images_folder': meta['images_folder'],
                # 'audio_file': meta['audio_file'],
                'conversations': [
                    {
                        'from': 'human',
                        'value': "<video>\n" + random.choice(video_questions)
                    },
                    {
                        'from': 'assistant',
                        'value': meta['original_metadata']['polish_caption']
                    }
                ]
            }
            
            # process audio metadata
            meta_processed_audio = {
                'id': meta['video_id'],
                # 'video_path': meta['original_metadata']['video_path'],
                # 'images_folder': meta['images_folder'],
                'audio_file': meta['audio_file'],
                'conversations': [
                    {
                        'from': 'human',
                        'value': "<audio>\n" + random.choice(audio_questions)
                    },
                    {
                        'from': 'assistant',
                        'value': meta['original_metadata']['music_caption'][0]['text']
                    }
                ]
            }

            meta_processed_video_audio = {
                'id': meta['video_id'],
                'video_path': meta['original_metadata']['video_path'],
                'images_folder': meta['images_folder'],
                'audio_file': meta['audio_file'],
                'conversations': [
                    {
                        'from': 'human',
                        'value': "<audio>\n<video>\n" + random.choice(audio_questions) # TODO: change this to video + audio question
                    },
                    {
                        'from': 'assistant',
                        'value': meta['original_metadata']['music_caption'][0]['text']
                    }
                ]
            }
            return meta_processed_video, meta_processed_audio, meta_processed_video_audio
        
        except Exception as e:
            print(f'Error processing metadata: {meta["video_id"]}')
            print(e)
            return None, None
        
    
    # process each split
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_root_dir, split)
        metas_dir = os.path.join(split_dir, 'metas')
        metas_processed_video_fp = os.path.join(split_dir, 'metas_video_convs.json')
        metas_processed_audio_fp = os.path.join(split_dir, 'metas_audio_convs.json')
        metas_processed_video_audio_fp = os.path.join(split_dir, 'metas_video_audio_convs.json')
        if not os.path.exists(metas_dir):
            print(f'No metadata found for [{split}] split.')
            continue
        
        print(f'Processing [{split}] split...')
        # process each meta file
        metas_processed_video_list = []
        metas_processed_audio_list = []
        metas_processed_video_audio_list = []
        for meta_file in os.listdir(metas_dir):
            with open(os.path.join(metas_dir, meta_file), 'r') as f:
                meta = json.load(f)
                
            # preprocess metadata and save to corresponding directory
            meta_processed_video, meta_processed_audio, meta_processed_video_audio = preprocess(meta)
            if meta_processed_video is not None:
                metas_processed_video_list.append(meta_processed_video)
            if meta_processed_audio is not None:
                metas_processed_audio_list.append(meta_processed_audio)
            if meta_processed_video_audio is not None:
                metas_processed_video_audio_list.append(meta_processed_video_audio)
                
        # save processed metadata
        with open(metas_processed_video_fp, 'w') as f:
            json.dump(metas_processed_video_list, f, indent=4)
        with open(metas_processed_audio_fp, 'w') as f:
            json.dump(metas_processed_audio_list, f, indent=4)
        with open(metas_processed_video_audio_fp, 'w') as f:
            json.dump(metas_processed_video_audio_list, f, indent=4)
            
    print('Preprocessing completed.')
    
    
if __name__ == '__main__':
    preprocess_MMTrail_metadata(MMTRAIL_DIR)
