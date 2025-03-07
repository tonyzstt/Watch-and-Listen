import os
import json
import copy
import torch
import soundfile as sf
from typing import Dict, Optional, Sequence, List
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

from conversation import get_conv_template, SeparatorStyle
from constant import *
from vision.processor import BaseProcessor, ImageEvalProcessor
from audio.mert_encoder import MERTEncoder
from utils import tokenizer_image_token, tokenizer_audio_token, tokenizer_image_audio_token

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
    mm_use_im_start_end: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pass


def preprocess_multimodal(
    source: List[Dict[str, str]],
    data_args: DataArguments,
    image_token_num=1,  # Set to 1 as place holder
    audio_token_num=1   # Set to 1 as place holder
) -> Dict:
    """Prepare input token template with multimodal placeholder"""
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return source
    
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

        # TODO: fix this
        # if data_args.mm_use_im_start_end:
        # We need the start video and start audio special tokens
        if True:
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

    return source


def preprocess(
    source: List[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False
) -> Dict:
    # TODO: Change this
    has_audio = False
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
    # The goal here is to tokenize all text that are not image/video/audio tokens
    # TODO: this can actually be merged
    if has_image and not has_audio:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif not has_image and has_audio:
        input_ids = torch.stack([tokenizer_audio_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_image and has_audio:
        input_ids = torch.stack([tokenizer_image_audio_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True
        ).input_ids

    targets = input_ids.clone()

    # TODO: Check if this will affect much, seems different mdoel use different sep style
    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    targets = input_ids.clone()
    
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
            # Here we use -4/-8 since we have both image and audio, we also need to mask vid/aud start/end
            if has_image and has_audio:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 8
            elif has_image or has_audio:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 4



            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        # print(target)
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(tokenizer.decode(z))
            exit()

        
        if total_len < tokenizer.model_max_length:
            pad_length = tokenizer.model_max_length - total_len
            # Pad input_ids with tokenizer.pad_token_id.
            input_ids = torch.nn.functional.pad(
                input_ids, (0, pad_length), value=tokenizer.pad_token_id
            )
            # Pad labels with -100.
            target = torch.nn.functional.pad(
                target, (0, pad_length), value=tokenizer.pad_token_id
            )

    return dict(
        input_ids=input_ids,
        labels=target,
        attention_mask=target.ne(tokenizer.pad_token_id),
    )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length].squeeze(1)
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'images' in instances[0]:
            images = [instance['images'] for instance in instances]

            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images

            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch



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
        
        raw_data = self.raw_data[idx]
        
        sources = preprocess_multimodal(
            copy.deepcopy(raw_data['conversations']),
            self.data_args
        )
        
        # TODO: consider add a 'video' section that load video information, and returns the number of visual tokens
        images = []
        if self.data_args.has_image:
            assert 'images_folder' in raw_data, "Image folder not found in data"
            
            image_folder = os.path.join(self.data_args.data_folder, raw_data['images_folder'])
            image_processor = self.image_processor
            
            if not os.path.exists(image_folder):
                print(f"Image folder {image_folder} not found")
            else:
                # all images in the folder
                num = 0
                for image_file_name in os.listdir(image_folder):

                    # TODO: fix this
                    if num == 10:
                        break
                    num += 1 
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
                
        if self.data_args.has_audio:
            assert 'audio_file' in raw_data, "Audio file not found in data"
            
            audio_processor = self.audio_processor
            audio_file_name = os.path.join(self.data_args.data_folder, raw_data['audio_file'])
            audio = audio_processor.get_hidden_states(audio_file_name)
        

        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=self.data_args.has_image,
            has_audio=self.data_args.has_audio
        )

        if self.data_args.has_image:
            # TODO: Sample fixed number of images, or consider padding
            data_dict['images'] = images[:5]
        if self.data_args.has_audio:
            data_dict['audio'] = audio
            
        return data_dict
    
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, image_processor, audio_processor) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(meta_data_path=data_args.meta_file_path, tokenizer=tokenizer, data_args=data_args, image_processor=image_processor, audio_processor=audio_processor)
    return train_dataset
        

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
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN], special_tokens=True)
    image_processor = ImageEvalProcessor()
    audio_processor = MERTEncoder()
    dataset = get_dataset(data_args, tokenizer, image_processor, audio_processor)
    
    data_0 = dataset[10]
    print(data_0.keys())
    print(data_0['images'][0].shape)
    print("input_ids:", data_0['input_ids'])
    print("labels:", data_0['labels'])
    print("attention_mask:", data_0['attention_mask'])
    print("images.size():", len(data_0['images']))
    print("audio.size():", len(data_0['audio']))
    print("Done!")
    