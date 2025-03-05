import os
import json
import copy
from typing import Optional, List, Dict
from dataclasses import dataclass, field

import torch
import psutil
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

from conversation import get_conv_template

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
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    is_multimodal: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pass


def preprocess_multimodal(
    sources: List[Dict[str, str]],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    
    # TODO: Implement multimodal preprocessing
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
    
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        
        print("Formatting inputs...Skip in lazy mode")
        raw_data : List[dict] = json.load(open(data_path, 'r'))
        
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.data_args = data_args
        # self.cached_data_dict = {}
        
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        assert idx < len(self), f"Index {idx} out of range"
        # if idx in self.cached_data_dict:
        #     return self.cached_data_dict[idx]
        
        raw_data = self.raw_data[idx]
        
        # FIXME: Currently there is no indicative token for multimodal data
        sources = preprocess_multimodal(
            copy.deepcopy(raw_data['conversations']),
            self.data_args
        )
        
        # if 'image' in raw_data:
        #     image_file_name = raw_data['image']
        #     image_folder = self.data_args.image_folder
        #     image_processor = self.data_args.image_processor
        #     image = Image.open(os.path.join(image_folder, image_file_name)).convert('RGB')
            
        #     # Padding and processing image
        #     if self.data_args.image_aspect_ratio == 'pad':
        #         def expand2square(pil_img, background_color):
        #             width, height = pil_img.size
        #             if width == height:
        #                 return pil_img
        #             elif width > height:
        #                 result = Image.new(pil_img.mode, (width, width), background_color)
        #                 result.paste(pil_img, (0, (width - height) // 2))
        #                 return result
        #             else:
        #                 result = Image.new(pil_img.mode, (height, height), background_color)
        #                 result.paste(pil_img, ((height - width) // 2, 0))
        #                 return result
        #         image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        #         image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        #     else:
        #         image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                
        # if 'audio' in sources:
        #     audio_file_name = sources['audio']
        #     audio_folder = self.data_args.audio_folder
        #     audio_processor = self.data_args.audio_processor
        #     audio_file_name = os.path.join(audio_folder, audio_file_name)
        #     audio = audio_processor.get_hidden_states(audio_file_name)
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image='image' in sources,
            has_audio='audio' in sources
        )
        
        # if 'image' in raw_data:
        #     data_dict['image'] = image
        # if 'audio' in raw_data:
        #     data_dict['audio'] = audio
            
        return data_dict
        

def get_dataset(
    data_args: DataArguments,
    tokenizer: transformers.PreTrainedTokenizer
) -> LazySupervisedDataset:
    return LazySupervisedDataset(
        data_args.data_path,
        tokenizer,
        data_args
    )
    

if __name__ == "__main__":
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args = [
            "--model_name_or_path", "/home/saberwu2002/CS229-Project/hf_ckp/vicuna-7b-v1.5",
            "--data_path", "/home/saberwu2002/CS229-Project/local_data/MMTrail_processed/test/metas_video_convs.json",
            "--output_dir", "/home/saberwu2002/CS229-Project/output/",
        ]
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    dataset = get_dataset(data_args, tokenizer)
    print(dataset[0])
    print(dataset[1])
    
