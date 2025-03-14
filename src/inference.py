import torch
torch.cuda.empty_cache()
from peft import PeftModel

import transformers
from transformers import (
    AutoTokenizer,
)

from constant import *
from train import MultiModalDataset, MultiModalLlama
from vision.processor import ImageEvalProcessor
from vision.clip_encoder import CLIPVisionTower
from audio.mert_encoder import MERTEncoder
from dataset import make_supervised_data_module, ModelArguments, DataArguments, TrainingArguments, DataCollatorForSupervisedDataset

if __name__ == "__main__":

    parser = transformers.HfArgumentParser(
                (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_path = model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.model_max_length = training_args.max_token_length
    num_added = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN], special_tokens=True)

    image_processor = ImageEvalProcessor()
    audio_processor = MERTEncoder()
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, image_processor=image_processor, audio_processor=audio_processor)
    llama_model_name = model_args.model_name_or_path
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = MultiModalDataset(data_module)
    collector = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    clip_model_name="openai/clip-vit-base-patch32"
    clip_vision_tower = CLIPVisionTower(clip_model_name).cuda()
    model = MultiModalLlama(llama_model_name=llama_model_name, vision_tower=clip_vision_tower, tokenizer=tokenizer, use_lora=False)
    model.llama.resize_token_embeddings(len(tokenizer))
    model.llama.config.vocab_size = len(tokenizer)
    PeftModel.from_pretrained(model.llama, model_args.pretrain_path, mean_resizing=False, output_loading_info=True)

    model.half()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():

        for data in data_module:

            input_ids = data['input_ids'].cuda()
            image = data['images']
            images = torch.stack(image, dim=0).cuda().half()
            outputs = model.generate(input_ids, images)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print("Generated Text:")
            print(generated_text)
