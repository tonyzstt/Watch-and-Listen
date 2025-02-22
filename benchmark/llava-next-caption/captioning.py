import os
import json

import torch
import numpy as np
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import PIL
import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def get_dataset(root_dir_data: str, split: str, sample: int = None):
    '''
    Get the dataset from the given root directory.
    
    Args:
        root_dir_data (str): Root directory of the dataset.
        split (str): Split of the dataset.
        sample (int): Number of frames to sample from each video.
        
    Returns:
        Dict[str, Dict[str, Any]]: Dataset.
    '''
    
    def get_captions_meta(root_dir_data, split, id):
        '''
        Get captions from the metadata file.
        
        Args:
            root_dir_data (str): Root directory of the dataset.
            split (str): Split of the dataset.
            id (str): Video id.
            
        Returns:
            Dict[str, str]: Dictionary of captions.
        '''
        
        meta_file = os.path.join(root_dir_data, split, 'metas', f'{id}.json')
        if not os.path.exists(meta_file):
            return None
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            captions = {
                'frame_caption': meta['original_metadata']['frame_caption'],
                'music_caption': meta['original_metadata']['music_caption'],
                'caption': meta['original_metadata']['caption'],
                'polish_caption': meta['original_metadata']['polish_caption']
            }
            
            print("Successfully loaded captions, id: ", id)
            
            return captions
    
    
    def load_frames(root_dir_data, split, id, sample=None):
        
        '''
        Load all frames of a video.
        
        Args:
            root_dir_data (str): Root directory of the dataset.
            split (str): Split of the dataset.
            id (str): Video id.
            
        Returns:
            np.ndarray: np array of frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        images_dir = os.path.join(root_dir_data, split, 'images', id)
        if not os.path.exists(images_dir):
            return None
        for image_file in os.listdir(images_dir):
            # read jpg image
            image = PIL.Image.open(os.path.join(images_dir, image_file))
            frames.append(np.array(image))
            
        if len(frames) == 0:
            return None
        
        frames = np.stack(frames)
        
        if sample is not None:
            # sample "sample" frames from the video
            indices = np.linspace(0, frames.shape[0] - 1, sample).astype(int)
            frames = frames[indices]
            
        # put to device
        frames = torch.tensor(frames).to(device)
        
        print("Successfully loaded frames, id: ", id)
        return frames
    
    
    # Get all video ids from test.json
    id_file = os.path.join(root_dir_data, f'{split}.json')
    assert os.path.exists(id_file), f'{id_file} does not exist'
    video_ids = []
    with open(id_file, 'r') as f:
        video_ids = json.load(f)
    
    dataset = {}
    print(f'Loading {len(video_ids)} videos')
    for id in tqdm.tqdm(video_ids):
        images = load_frames(root_dir_data, split, id, sample=sample)
        if images is None:
            continue
        
        captions = get_captions_meta(root_dir_data, split, id)
        # TODO: video and audio
        if captions is not None:
            dataset[id] = {
                'captions': captions,
                'images': images
            }
            
    print(f'Loaded {len(dataset)} videos')
            
    return dataset


def load_llava_next_model(model_path: str):
    '''
    Load the model from the given path.
    
    Args:
        model_path (str): Path to the model.
        
    Returns:
        torch.nn.Module: Model.
    '''
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = LlavaNextVideoProcessor.from_pretrained(model_path)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map='auto'
    )
    
    model.to(device)
    
    return model, processor


def generate_caption_video(model, processor, dataset):
    '''
    Generate captions for videos in the given dataset.
    
    Args:
        model (torch.nn.Module): Model.
        processor (LlavaNextVideoProcessor): Processor.
        dataset (Dict[str, Dict[str, Any]]): Dataset.
        
    Returns:
        Dict[str, str]: Dictionary of captions.
    '''
    
    conversation_video = [
        {
            "role": "user",
            "content": [
                    {"type": "text", "text": "Describe the video."},
                    {"type": "video"},
                ],
        },
    ]
    
    prompt = processor.apply_chat_template(conversation_video, add_generation_prompt=True)
    print(prompt)
    
    outputs = {}
    
    for id in dataset:
        inputs = processor([prompt], videos=[dataset[id]['images']], padding=True, return_tensors="pt").to(model.device)
        generate_kwargs = {"max_new_tokens": 50, "do_sample": True, "top_p": 0.9}
        output = model.generate(**inputs, **generate_kwargs)
        generated_text = processor.decode(output[0], skip_special_tokens=True)
        # get only output from the assistant
        generated_text = generated_text.split('ASSISTANT: ')[-1]
        print(generated_text)
        outputs[id] = generated_text
        
    return outputs



    

def main():
    root_dir_data = '/home/saberwu2002/disk-data/data/MMTrail_processed'
    split = 'test'
    sample = 30
    dataset = get_dataset(root_dir_data, split, sample)
    model_path = '/home/saberwu2002/disk-data/checkpoints/llava-next-video-7b-hf'
    model, processor = load_llava_next_model(model_path)
    captions = generate_caption_video(model, processor, dataset)
    
    # save to json
    with open('captions.json', 'w') as f:
        json.dump(captions, f)
        
        
if __name__ == '__main__':
    main()