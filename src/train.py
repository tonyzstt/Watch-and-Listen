import torch
import torch.nn as nn
from torch.utils.data import Dataset
torch.cuda.empty_cache()
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer
)
from transformers.trainer_pt_utils import LabelSmoother
from huggingface_hub import login

from constant import *
from huggingface_hub import login
from vision.vision_projection import build_vision_projector
from vision.processor import ImageEvalProcessor
from vision.clip_encoder import CLIPVisionTower
from audio.audio_projection import build_audio_projector
from audio.mert_encoder import MERTEncoder
from config.config import VisionProjectorConfig, AudioProjectorConfig
from dataset import make_supervised_data_module, ModelArguments, DataArguments, TrainingArguments, DataCollatorForSupervisedDataset, IGNORE_TOKEN_ID


def print_gpu_memory_usage(device=0):
    allocated = torch.cuda.memory_allocated(device) / 1024**2  
    reserved = torch.cuda.memory_reserved(device) / 1024**2   
    print(f"GPU {device} - Allocated Memory: {allocated:.2f} MB")
    print(f"GPU {device} - Reserved Memory: {reserved:.2f} MB")
    
class MultiModalDataset(Dataset):
    """
    Modified dataset class that accepts a data module's output.
    Each example is assumed to have the following keys:
      - "input_ids": pre-tokenized input text.
      - "labels": ground truth labels (already processed).
      - "attention_mask": attention mask for the input.
      - "images": visual features (will be used as visual_tokens).
      - "audio": audio features (will be used as audio_tokens).
    """
    def __init__(self, data, max_length=512):
        """
        Args:
            data (list or Dataset): A collection of examples from your data module.
            max_length (int): Maximum sequence length (if any additional processing is needed).
                              (Not used in this version but kept for backward compatibility.)
        """
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_ids = item["input_ids"]
        labels = item["labels"]
        # attention_mask = item["attention_mask"] # FIXME: I don't think we need this here
        images = item["images"] if "images" in item else None
        audio = item["audio"] if "audio" in item else None

        return {
            "input_ids": input_ids,
            "labels": labels,
            # "attention_mask": attention_mask, # Not needed?
            "images": images,
            "audio": audio,
        }

class MultiModalLlama(nn.Module):
    def __init__(
        self,
        vision_tower=None,
        llama_model_name=None,
        vision_config_params=None,
        audio_config_params=None,
        use_lora=True,
    ):
        super(MultiModalLlama, self).__init__()

        self.llama = AutoModelForCausalLM.from_pretrained(llama_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vision_tower = vision_tower

        # Only focus on the lora parameters
        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj"],  
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama = get_peft_model(self.llama, lora_config)

        if vision_config_params is None:
            vision_config_params = {
                "mm_projector_type": "mlp2x_gelu",
                "mm_hidden_size": 768,
                "hidden_size": 4096,
            }
        if audio_config_params is None:
            audio_config_params = {
                "mm_projector_type": "mlp2x_gelu",
                "audio_hidden_size": 768,
                "hidden_size": 4096,
            }

        # Projection Layers
        self.vision_projector = build_vision_projector(
            VisionProjectorConfig(**vision_config_params)
        )
        self.audio_projector = build_audio_projector(
            AudioProjectorConfig(**audio_config_params)
        )

    def freeze_llama(self):
        """Freeze all parameters of the LLaMA model."""
        for param in self.llama.parameters():
            param.requires_grad = False

    def unfreeze_llama(self):
        """Unfreeze only the last layer of the LLaMA model."""
        for name, param in self.llama.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def freeze_projection_layers(self):
        """Freeze vision and audio projection layers."""
        for param in self.vision_projector.parameters():
            param.requires_grad = False
        for param in self.audio_projector.parameters():
            param.requires_grad = False

    def unfreeze_vision_projection_layers(self):
        """Unfreeze vision and audio projection layers."""
        for param in self.vision_projector.parameters():
            param.requires_grad = True

    def forward(self, input_ids, images=None, audio=None, labels=None):
        """
        Input:
          - input_ids: tokenized text (question+answer) of shape (batch, seq_len)
          - visual_tokens: visual features of shape (batch, v_len, feat_dim)
          - audio_tokens: audio features of shape (batch, a_len, feat_dim)
          - labels: token labels for LM loss.
        """

        embeddings = []
        new_labels = []
        if audio is None:

            is_video = False
            if input_ids.shape[0] != images.shape[0]:
                is_video = True
            
            image_features = self.vision_tower(images)
            visual_embeds = self.vision_projector(image_features) 

            if is_video:
                visual_embeds = visual_embeds.view(input_ids.shape[0], -1, *visual_embeds.shape[1:])

            for input_id, label, visual_embed in zip(input_ids, labels, visual_embeds):

                visual_embed = visual_embed.view(-1, visual_embed.shape[-1])
                
                # TODO: remove this hardcode
                indices_32002 = (input_id == 32002).nonzero(as_tuple=True)[0]
                first_index_32002 = indices_32002[0].item() if indices_32002.numel() > 0 else None
                indices_32003 = (input_id == 32003).nonzero(as_tuple=True)[0]
                first_index_32003 = indices_32003[0].item() if indices_32003.numel() > 0 else None

                seq_len = int(input_id.ne(tokenizer.pad_token_id).sum())
                target_masked_len = (label == IGNORE_TOKEN_ID).sum()
                valid_label = label[target_masked_len:seq_len]
                if False:  # Inspect and check the correctness of masking
                    z = valid_label.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                    print(tokenizer.decode(z))
                    exit()
                prefix_id = input_id[:first_index_32002+1]
                suffix_id = input_id[first_index_32003:seq_len]
                prefix_embeds = self.llama.get_input_embeddings()(prefix_id)  
                suffix_embeds = self.llama.get_input_embeddings()(suffix_id)  
                # -1 for we remove the image token place holder
                new_target_mask_len = target_masked_len - 1 + visual_embed.shape[0]
                input_embds = torch.cat([prefix_embeds, visual_embed, suffix_embeds], dim=0)
                pad_embds = torch.zeros((tokenizer.model_max_length - input_embds.shape[0], input_embds.shape[-1])).cuda().to(torch.float16)
                input_embds = torch.cat([input_embds, pad_embds], dim=0)
                pad_length = tokenizer.model_max_length - new_target_mask_len - len(valid_label)
                new_label = torch.cat([torch.full((new_target_mask_len,), tokenizer.unk_token_id).cuda(), valid_label, torch.full((pad_length,), tokenizer.pad_token_id).cuda()], dim=0)
                if False:  # Inspect and check the correctness of masking
                    z = new_label.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                    print(tokenizer.decode(z))
                    exit()
                
                embeddings.append(input_embds)
                new_labels.append(new_label)
        
        elif images is None:
            
            audio_embeds = self.audio_projector(audio)
            audio_embeds = audio_embeds.view(audio_embeds.shape[0], -1, audio_embeds.shape[-1])

            for input_id, label, audio_embed in zip(input_ids, labels, audio_embeds):
                
                # TODO: remove this hardcode
                indices_32004 = (input_id == 32004).nonzero(as_tuple=True)[0]
                first_index_32004 = indices_32004[0].item() if indices_32004.numel() > 0 else None
                indices_32005 = (input_id == 32005).nonzero(as_tuple=True)[0]
                first_index_32005 = indices_32005[0].item() if indices_32005.numel() > 0 else None

                seq_len = int(input_id.ne(tokenizer.pad_token_id).sum())
                target_masked_len = (label == IGNORE_TOKEN_ID).sum()
                valid_label = label[target_masked_len:seq_len]
                if False:  # Inspect and check the correctness of masking
                    z = valid_label.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                    print(tokenizer.decode(z))
                    exit()
                prefix_id = input_id[:first_index_32004+1]
                suffix_id = input_id[first_index_32005:seq_len]
                prefix_embeds = self.llama.get_input_embeddings()(prefix_id)  
                suffix_embeds = self.llama.get_input_embeddings()(suffix_id)  
                # -1 for we remove the image token place holder
                new_target_mask_len = target_masked_len - 1 + audio_embed.shape[0]
                input_embds = torch.cat([prefix_embeds, audio_embed, suffix_embeds], dim=0)
                pad_embds = torch.zeros((tokenizer.model_max_length - input_embds.shape[0], input_embds.shape[-1])).cuda().to(torch.float16)
                input_embds = torch.cat([input_embds, pad_embds], dim=0)
                pad_length = tokenizer.model_max_length - new_target_mask_len - len(valid_label)
                new_label = torch.cat([torch.full((new_target_mask_len,), tokenizer.unk_token_id).cuda(), valid_label, torch.full((pad_length,), tokenizer.pad_token_id).cuda()], dim=0)
                if False:  # Inspect and check the correctness of masking
                    z = new_label.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                    print(tokenizer.decode(z))
                    exit()
                
                embeddings.append(input_embds)
                new_labels.append(new_label)

        else:

            is_video = False
            if input_ids.shape[0] != images.shape[0]:
                is_video = True
            
            image_features = self.vision_tower(images)
            visual_embeds = self.vision_projector(image_features) 

            if is_video:
                visual_embeds = visual_embeds.view(input_ids.shape[0], -1, *visual_embeds.shape[1:])

            audio_embeds = self.audio_projector(audio)
            audio_embeds = audio_embeds.view(audio_embeds.shape[0], -1, audio_embeds.shape[-1])

            for input_id, label, audio_embed, visual_embed in zip(input_ids, labels, audio_embeds, visual_embeds):

                visual_embed = visual_embed.view(-1, visual_embed.shape[-1])
                
                # TODO: remove this hardcode
                # For vision
                indices_32002 = (input_id == 32002).nonzero(as_tuple=True)[0]
                first_index_32002 = indices_32002[0].item() if indices_32002.numel() > 0 else None
                indices_32003 = (input_id == 32003).nonzero(as_tuple=True)[0]
                first_index_32003 = indices_32003[0].item() if indices_32003.numel() > 0 else None
                
                # TODO: remove this hardcode
                # For Audio
                indices_32004 = (input_id == 32004).nonzero(as_tuple=True)[0]
                first_index_32004 = indices_32004[0].item() if indices_32004.numel() > 0 else None
                indices_32005 = (input_id == 32005).nonzero(as_tuple=True)[0]
                first_index_32005 = indices_32005[0].item() if indices_32005.numel() > 0 else None

                seq_len = int(input_id.ne(tokenizer.pad_token_id).sum())
                target_masked_len = (label == IGNORE_TOKEN_ID).sum()
                valid_label = label[target_masked_len:seq_len]
                if False:  # Inspect and check the correctness of masking
                    z = valid_label.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                    print(tokenizer.decode(z))
                    exit()
                prefix_id = input_id[:first_index_32002+1]
                midfix_id = input_id[first_index_32003:first_index_32004+1]
                suffix_id = input_id[first_index_32005:seq_len]
                prefix_embeds = self.llama.get_input_embeddings()(prefix_id)  
                midfix_embeds = self.llama.get_input_embeddings()(midfix_id)  
                suffix_embeds = self.llama.get_input_embeddings()(suffix_id)  
                # -1 for we remove the image and audio token place holder
                new_target_mask_len = target_masked_len - 2 + audio_embed.shape[0] + visual_embed.shape[0]
                input_embds = torch.cat([prefix_embeds, visual_embed, midfix_embeds, audio_embed, suffix_embeds], dim=0)
                pad_embds = torch.zeros((tokenizer.model_max_length - input_embds.shape[0], input_embds.shape[-1])).cuda().to(torch.float16)
                input_embds = torch.cat([input_embds, pad_embds], dim=0)
                pad_length = tokenizer.model_max_length - new_target_mask_len - len(valid_label)
                new_label = torch.cat([torch.full((new_target_mask_len,), tokenizer.unk_token_id).cuda(), valid_label, torch.full((pad_length,), tokenizer.pad_token_id).cuda()], dim=0)
                if False:  # Inspect and check the correctness of masking
                    z = new_label.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                    print(tokenizer.decode(z))
                    exit()
                
                embeddings.append(input_embds)
                new_labels.append(new_label)

        embeddings = torch.stack(embeddings, dim=0)
        new_labels = torch.stack(new_labels, dim=0)
        new_attention_mask = torch.tensor(new_labels != tokenizer.pad_token_id).cuda().to(torch.int64)

        outputs = self.llama(
            inputs_embeds=embeddings,
            attention_mask=new_attention_mask,
            labels=new_labels,
        )

        # print_gpu_memory_usage()

        return outputs

    @property
    def config(self):
        return self.llama.config

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.llama.prepare_inputs_for_generation(input_ids, **kwargs)

if __name__ == "__main__":

    parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # TODO: Implement fp8 to enable longer length and larger batch size
    tokenizer.model_max_length = 1024
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN], special_tokens=True)
    image_processor = ImageEvalProcessor()
    audio_processor = MERTEncoder()
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, image_processor=image_processor, audio_processor=audio_processor)

    login(token='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')
    llama_model_name = model_args.model_name_or_path
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = MultiModalDataset(data_module)
    collector = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    clip_model_name="openai/clip-vit-base-patch32"
    # audio_model_name="facebook/wav2vec2-base-960h"
    clip_vision_tower = CLIPVisionTower(clip_model_name).cuda()
    model = MultiModalLlama(llama_model_name=llama_model_name, vision_tower=clip_vision_tower)
    model.llama.resize_token_embeddings(len(tokenizer))
    id = tokenizer(DEFAULT_VID_START_TOKEN).input_ids
    prefix_embeds = model.llama.get_input_embeddings()(torch.tensor(id))  
    # audio_encoder = AudioEncoder(audio_model_name)

    # Stage 1 of training
    model.freeze_llama()               
    model.unfreeze_vision_projection_layers()    
    training_args_stage1 = TrainingArguments(
        output_dir="./llama_finetuned_stage1",
        per_device_train_batch_size=1,
        num_train_epochs=20,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        save_safetensors=False,
        remove_unused_columns=False,  
        deepspeed="deepspeed.json",
    )
    trainer_stage1 = Trainer(
        model=model,
        args=training_args_stage1,
        train_dataset=dataset,
        data_collator=collector
    )

    print("=== Stage 1: Training Projection Layers Only ===")
    trainer_stage1.train()
    # Stage 2 of training
    model.freeze_projection_layers()  
    model.unfreeze_llama()           
    training_args_stage2 = TrainingArguments(
        output_dir="./llama_finetuned_stage2",
        per_device_train_batch_size=1,
        num_train_epochs=20,
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        save_safetensors=False,
        remove_unused_columns=False,
        deepspeed="deepspeed.json"
    )
    trainer_stage2 = Trainer(
        model=model,
        args=training_args_stage2,
        train_dataset=dataset,
        data_collator=collector
    )

    print("=== Stage 2: Fine-Tuning LLaMA Only (Projection Layers Frozen) ===")
    trainer_stage2.train()