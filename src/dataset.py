import os
import json
import copy
import torch
import soundfile as sf
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from PIL import Image

from constant import *

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                            metadata={"help": "Path to the data directory."})
    # lazy_preprocessing: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    audio_folder: Optional[str] = field(default=None)
    # image_aspect_ratio: str = 'square'


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
    sources: List[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False
) -> Dict:
    # FIXME: Implement get_conversation_template
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["role"]] != conv.roles[0]:
            # Skip the first one if it's not from human
            source = source[1:]
            
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"Role mismatch at {i}"
                conv.append_message(role, sentence["text"])
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
    
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        
        print("Formatting inputs...Skip in lazy mode")
        raw_data : List[dict] = json.load(open(data_path, 'r'))
        
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.cached_data_dict = {}
        
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        assert idx < len(self), f"Index {idx} out of range"
        if idx in self.cached_data_dict:
            return self.cached_data_dict[idx]
        
        # FIXME: Currently there is no indicative token for multimodal data
        # TODO: This should happens after the processing steps below since we need to know the length of image/audio dataset
        sources = preprocess_multimodal(
            copy.deepcopy([self.raw_data[idx]['conversations']]),
            self.data_args
        )
        
        # TODO: consider add a 'video' section that load video information, and returns the number of visual tokens
        if 'image' in sources:
            image_file_name = sources['image']
            image_folder = self.data_args.image_folder
            image_processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file_name)).convert('RGB')
            
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
                image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image)['pixel_values'][0]
            else:
                image = image_processor.preprocess(image)['pixel_values'][0]
                
        if 'audio' in sources:
            audio_file_name = sources['audio']
            audio_folder = self.data_args.audio_folder
            audio_processor = self.data_args.audio_processor

            wav, sample_rate_ = sf.read(os.path.join(audio_folder, audio_file_name), dtype='float32', always_2d=True)
            audio = audio_processor.preprocess(wav, sample_rate_)['audio_wav']
            
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image='image' in sources,
            has_audio='audio' in sources
        )
        
        if 'image' in sources:
            data_dict['image'] = image
        if 'audio' in sources:
            data_dict['audio'] = audio
            
        return data_dict
        
        