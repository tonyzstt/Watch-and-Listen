import numpy as np
import os
import json
import tqdm
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import torch
import PIL


        

def get_dataset(root_dir_data: str, split: str, sample: int = None):
    """
    test/
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
    """
    
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
            image = PIL.Image.open(os.path.join(images_dir, image_file), format="rgb24")
            frames.append(np.array(image))
        frames = np.stack(frames)
        print(frames.shape)
        
        if sample is not None:
            # sample "sample" frames from the video
            indices = np.linspace(0, frames.shape[0] - 1, sample).astype(int)
            frames = frames[indices]
        
        return frames
    
    
    # Get all video ids from test.json
    id_file = os.path.join(root_dir_data, f'{split}.json')
    assert os.path.exists(id_file), f'{id_file} does not exist'
    video_ids = []
    with open(id_file, 'r') as f:
        video_ids = json.load(f)
    
    dataset = {}
    for id in video_ids:
        captions = get_captions_meta(root_dir_data, id)
        images = load_frames(root_dir_data, split, id, sample=sample)
        # TODO: video and audio
        if captions is not None and images is not None:
            dataset[id] = {
                'captions': captions,
                'images': images
            }
            
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

    processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        "llava-hf/LLaVA-NeXT-Video-7B-hf",
        quantization_config=quantization_config,
        device_map='auto'
    )
    
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
        generate_kwargs = {"max_new_tokens": 100, "do_sample": True, "top_p": 0.9}
        outputs = model.generate(**inputs, **generate_kwargs)
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)
        outputs[id] = generated_text
        
    return outputs
    

def main():
    root_dir_data = 'MMTrail'
    split = 'test'
    sample = 30
    dataset = get_dataset(root_dir_data, split, sample)
    model_path = 'llava-hf/LLaVA-NeXT-Video-7B-hf'
    model, processor = load_llava_next_model(model_path)
    captions = generate_caption_video(model, processor, dataset)
    
    # save to json
    with open('captions.json', 'w') as f:
        json.dump(captions, f)