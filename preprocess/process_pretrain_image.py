from typing import Tuple, Optional
import json
import os
import random
from tqdm import tqdm


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
CC3M_595K_DIR = "/home/tonyzst/Desktop/Chat-UniVi/data/CC3M-595K"
COCO_DIR = "/home/tonyzst/Downloads/train2014"
COCO_JSON = "/home/tonyzst/Downloads/coco_cap_chat.json"
CC3M_JSON = "/home/tonyzst/Downloads/chat.json"
QUESTIONS_DIR = "/home/tonyzst/Desktop/CS229-Project/preprocess"


def load_questions(root_dir: str) -> Tuple[list, list]:
    """
    Load random questions for video and audio captioning
    """
    image_questions = []
    with open(os.path.join(root_dir, 'alignment_questions_image.txt')) as f:
        for line in f:
            image_questions.append(line.strip())
    
    return image_questions


def preprocess_image_metadata():
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
    
    def preprocess(meta: dict, dir: str) -> Optional[Tuple[dict]]:
        """
        Retrieve necessary information from metadata and construct conversation pairs
        """

        image_file = meta['image']
        image_file_names = os.path.join(dir, image_file)
        
        try:
            # process video metadata
            meta_processed_image = {
                'id': meta['id'],
                'images_folder': image_file_names,
                'conversations': [
                    {
                        'from': 'human',
                        'value': meta["conversations"][0]["value"]
                    },
                    {
                        'from': 'assistant',
                        'value': meta['conversations'][1]['value']
                    }
                ]
            }
            
            return meta_processed_image
        
        except Exception as e:
            print(f'Error processing metadata: {meta["video_id"]}')
            print(e)
            return None, None

    
    metas_processed_video_fp = 'metas_image_convs.json'

    metas_processed_image_list = []
    with open(CC3M_JSON, 'r') as f:
            metas = json.load(f)

    for meta in tqdm(metas):


        image_file = meta['image']
        image_path = os.path.join(CC3M_595K_DIR, image_file)
        if not os.path.exists(image_path):
             continue

        meta_processed_image = preprocess(meta, CC3M_595K_DIR)
        if meta_processed_image is not None:
            metas_processed_image_list.append(meta_processed_image)

    with open(COCO_JSON, 'r') as f:
            metas = json.load(f)

    for meta in tqdm(metas):



        image_file = meta['image']
        image_path = os.path.join(COCO_DIR, image_file)
        if not os.path.exists(image_path):
             continue
        
        meta_processed_image = preprocess(meta, COCO_DIR)
        if meta_processed_image is not None:
            metas_processed_image_list.append(meta_processed_image)

    random.shuffle(metas_processed_image_list)

    # save processed metadata
    with open(metas_processed_video_fp, 'w') as f:
        json.dump(metas_processed_image_list[:50000], f, indent=4)
            
    print('Preprocessing completed.')
    
    
if __name__ == '__main__':
    preprocess_image_metadata()
