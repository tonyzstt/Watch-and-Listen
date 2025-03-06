import os
import json
import copy
import torch
import soundfile as sf
from typing import Optional, List, Dict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import psutil
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from huggingface_hub import login

from conversation import get_conv_template
from constant import *
from vision.processor import BaseProcessor, ImageEvalProcessor
from audio.mert_encoder import MERTEncoder

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
CONV_TEMPLATE_NAME = 'vicuna_v1.1'


# Check device and memory
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
ram_gb = psutil.virtual_memory().total / 1e9
print(f"RAM: {ram_gb} GB")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/saberwu2002/CS229-Project/hf_ckp/vicuna-7b-v1.5")


@dataclass
class DataArguments:
    meta_file_path: str = field(default=None, metadata={"help": "Path to the meta data json file."})
    data_folder: str = field(default=None, metadata={"help": "Path to the data folder."})
    is_multimodal: bool = field(default=False)
    has_video: bool = field(default=False)
    has_image: bool = field(default=False)
    has_audio: bool = field(default=False)
    image_aspect_ratio: str = field(default=None, metadata={"help": "Aspect ratio of the image."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pass


def preprocess_multimodal(
    sources: List[Dict[str, str]],
    data_args: DataArguments,
    image_token_num=1,  # For number of image tokens
    audio_token_num=1   # For number of audio tokens
) -> Dict:
    """Prepare input token template with multimodal placeholder"""
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    
    for source in sources:
        for sentence in source:
            if (DEFAULT_IMAGE_TOKEN in sentence['value'] or 
                DEFAULT_VIDEO_TOKEN in sentence['value'] or 
                DEFAULT_AUDIO_TOKEN in sentence['value']):
                
                # Image Tokens
                sentence['value'] = sentence['value'].replace(
                    DEFAULT_IMAGE_TOKEN + '\n', DEFAULT_IMAGE_TOKEN
                ).strip()
                sentence['value'] = sentence['value'].replace(
                    '\n' + DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN
                ).strip()
                if sentence['value'].endswith(DEFAULT_IMAGE_TOKEN):
                    IMAGE_TOKEN_NUM = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM, ''
                    ).strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                
                # Video Tokens
                if sentence['value'].endswith(DEFAULT_VIDEO_TOKEN):
                    VIDEO_TOKEN_NUM = sentence['value'].count(DEFAULT_VIDEO_TOKEN)
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM, ''
                    ).strip()
                    sentence['value'] = DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                
                # Audio Tokens
                sentence['value'] = sentence['value'].replace(
                    DEFAULT_AUDIO_TOKEN + '\n', DEFAULT_AUDIO_TOKEN
                ).strip()
                sentence['value'] = sentence['value'].replace(
                    '\n' + DEFAULT_AUDIO_TOKEN, DEFAULT_AUDIO_TOKEN
                ).strip()
                if sentence['value'].endswith(DEFAULT_AUDIO_TOKEN):
                    AUDIO_TOKEN_NUM = sentence['value'].count(DEFAULT_AUDIO_TOKEN)
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_AUDIO_TOKEN * AUDIO_TOKEN_NUM, ''
                    ).strip()
                    sentence['value'] = DEFAULT_AUDIO_TOKEN * AUDIO_TOKEN_NUM + sentence['value']
                    sentence['value'] = sentence['value'].strip()

                # TODO: conversation
                # if "mmtag" in conversation_lib.default_conversation.version:
                #     sentence['value'] = sentence['value'].replace(
                #         DEFAULT_IMAGE_TOKEN,
                #         '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>'
                #     )
                
                IMAGE_TOKEN_NUM = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                if IMAGE_TOKEN_NUM > MAX_IMAGE_LENGTH:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM,
                        DEFAULT_IMAGE_TOKEN * MAX_IMAGE_LENGTH
                    ).strip()

            # Prepare replacement tokens for image, video, and audio
            replace_token = DEFAULT_IMAGE_TOKEN
            vid_replace_token = DEFAULT_IMAGE_TOKEN * image_token_num
            aud_replace_token = DEFAULT_AUDIO_TOKEN * audio_token_num
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                vid_replace_token = DEFAULT_VID_START_TOKEN + vid_replace_token + DEFAULT_VID_END_TOKEN
                aud_replace_token = DEFAULT_AUDIO_START_TOKEN + aud_replace_token + DEFAULT_AUDIO_END_TOKEN

            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token + '\n'
            )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_VIDEO_TOKEN, vid_replace_token + '\n'
            )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_AUDIO_TOKEN, aud_replace_token + '\n'
            )
            sentence['value'] = sentence['value'].replace('\n\n', '\n')
    return sources


def preprocess(
    source: List[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False
) -> Dict:
    # FIXME: Implement get_conversation_template
    conv = get_conv_template(CONV_TEMPLATE_NAME)
    roles = {"human": conv.roles[0], "assistant": conv.roles[1]}
    
    # Apply prompt templates
    conversations = []
    if roles[source[0]["from"]] != conv.roles[0]:
        # Skip the first one if it's not from human
        source = source[1:]
        
    conv.messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"Role mismatch"
        conv.append_message(role, sentence["value"])
    conversations.append(conv.get_prompt())
            
    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True
    ).input_ids
    targets = input_ids.clone()
    
    # FIXME: Implement SeparatorStyle
    # assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
    
    # TODO: Handle multimodal data
    
    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class LazySupervisedDataset(Dataset):
    """Dataset for pretraining for multimodal feature alignment."""
    
    def __init__(self, meta_data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_processor: Optional[BaseProcessor],
                 audio_processor: Optional[nn.Module],
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        
        print("Formatting inputs...Skip in lazy mode")
        raw_data : List[dict] = json.load(open(meta_data_path, 'r'))
        
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.data_args = data_args
        
        if self.data_args.has_image:
            assert self.image_processor is not None, "Image processor not found"
        if self.data_args.has_audio:
            assert self.audio_processor is not None, "Audio processor not found"
        # self.cached_data_dict = {}
        
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        assert idx < len(self), f"Index {idx} out of range"
        # if idx in self.cached_data_dict:
        #     return self.cached_data_dict[idx]
        
        raw_data = self.raw_data[idx]
        
        # FIXME: Currently there is no indicative token for multimodal data
        # TODO: This should happens after the processing steps below since we need to know the length of image/audio dataset
        sources = preprocess_multimodal(
            copy.deepcopy(raw_data['conversations']),
            self.data_args
        )
        
        # TODO: consider add a 'video' section that load video information, and returns the number of visual tokens
        
        images = []
        if data_args.has_image:
            assert 'images_folder' in raw_data, "Image folder not found in data"
            
            root_folder = self.data_args.data_folder
            image_folder = os.path.join(root_folder, raw_data['images_folder'])
            image_processor = self.image_processor
            
            if not os.path.exists(image_folder):
                print(f"Image folder {image_folder} not found")
            else:
                # all images in the folder
                for image_file_name in os.listdir(image_folder):
                    image_fp = os.path.join(image_folder, image_file_name)
                    image = Image.open(image_fp).convert('RGB')
                
                    # Padding and processing image
                    if self.data_args.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        # FIXME: image_processor.image_mean is not defined
                        image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
                        image = image_processor.preprocess(image)
                    else:
                        image = image_processor.preprocess(image)
                        
                    images.append(image)
                
        if data_args.has_audio:
            assert 'audio_file' in raw_data, "Audio file not found in data"
            
            audio_file_name = raw_data['audio_file']
            audio_folder = self.data_args.audio_folder
            audio_processor = self.audio_processor
            audio_file_name = os.path.join(audio_folder, audio_file_name)
            audio = audio_processor.get_hidden_states(audio_file_name)
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=data_args.has_image,
            has_audio=data_args.has_audio
        )
        
        if data_args.has_image:
            data_dict['images'] = images
        if data_args.has_audio:
            data_dict['audio'] = audio
            
        return data_dict
        

def get_dataset(
    data_args: DataArguments,
    tokenizer: transformers.PreTrainedTokenizer,
    image_processor: Optional[BaseProcessor],
    audio_processor: Optional[nn.Module]
) -> LazySupervisedDataset:
    return LazySupervisedDataset(
        data_args.meta_file_path,
        tokenizer,
        image_processor,
        audio_processor,
        data_args
    )
    

if __name__ == "__main__":
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args = [
            "--model_name_or_path", "lmsys/vicuna-7b-v1.5",
            "--meta_file_path", "/home/saberwu2002/CS229-Project/local_data/MMTrail_processed/test/metas_video_convs.json",
            "--output_dir", "/home/saberwu2002/CS229-Project/output/",
            "--has_video",
            "--has_image",
            # "--has_audio",
            "--data_folder", "/home/saberwu2002/CS229-Project/local_data/MMTrail_processed/test/"
        ]
    )
    # print(model_args.model_name_or_path)
    # exit()
    login(token='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')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_auth_token=True, use_fast=False)
    image_processor = ImageEvalProcessor()
    # audio_processor = MERTEncoder() # TODO: fill in arguments
    audio_processor = None
    dataset = get_dataset(data_args, tokenizer, image_processor, audio_processor)
    
    data_0 = dataset[0]
    print("input_ids:", data_0['input_ids'])
    print("labels:", data_0['labels'])
    print("attention_mask:", data_0['attention_mask'])
    print("images.size():", len(data_0['images']))
    # print("audio.size():", len(data_0['audio']))
    print("Done!")
    
