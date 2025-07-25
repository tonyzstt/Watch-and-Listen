import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
torch.cuda.empty_cache()
from peft import LoraConfig, get_peft_model

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer
)

from constant import *
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
        images = item["images"] if "images" in item else None
        audio = item["audio"] if "audio" in item else None

        return {
            "input_ids": input_ids,
            "labels": labels,
            "images": images,
            "audio": audio,
        }

class MultiModalLlama(nn.Module):
    def __init__(
        self,
        vision_tower=None,
        tokenizer=None,
        llama_model_name=None,
        vision_config_params=None,
        audio_config_params=None,
        use_lora=True,
    ):
        super(MultiModalLlama, self).__init__()

        self.llama = AutoModelForCausalLM.from_pretrained(llama_model_name)
        self.tokenizer = tokenizer
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
            self.llama.print_trainable_parameters()

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

        self.vision_projector = build_vision_projector(
            VisionProjectorConfig(**vision_config_params)
        )
        self.audio_projector = build_audio_projector(
            AudioProjectorConfig(**audio_config_params)
        )

    @classmethod
    def from_pretrained(cls, base, load_directory, vision_tower=None):

        config_path = os.path.join(load_directory, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        use_lora = config.get("use_lora", False)
        vision_config_params = config.get("vision_config_params", None)
        audio_config_params = config.get("audio_config_params", None)

        llama_dir = os.path.join(load_directory, "llama")
        tokenizer = AutoTokenizer.from_pretrained(llama_dir)

        base_model = AutoModelForCausalLM.from_pretrained(base)

        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model_llama = get_peft_model(base_model, lora_config)
        else:
            model_llama = base_model

        adapter_path = os.path.join(llama_dir, "lora_params.bin")
        if os.path.exists(adapter_path):
            lora_state_dict = torch.load(adapter_path, map_location="cpu")
            current_state_dict = model_llama.state_dict()
            current_state_dict.update(lora_state_dict)
            model_llama.load_state_dict(current_state_dict)

        model = cls(
            vision_tower=vision_tower,
            llama_model_name=llama_model_name,
            vision_config_params=vision_config_params,
            audio_config_params=audio_config_params,
            use_lora=use_lora,
        )
        model.llama = model_llama
        model.tokenizer = tokenizer

        vision_projector_dir = os.path.join(load_directory, "vision_projector")
        vision_state = torch.load(
            os.path.join(vision_projector_dir, "pytorch_model.bin"),
            map_location="cpu"
        )
        model.vision_projector.load_state_dict(vision_state)

        audio_projector_dir = os.path.join(load_directory, "audio_projector")
        audio_state = torch.load(
            os.path.join(audio_projector_dir, "pytorch_model.bin"),
            map_location="cpu"
        )
        model.audio_projector.load_state_dict(audio_state)

        return model
    
    def save_vision_layer(self, save_path):
        """Save the vision projector's parameters to the specified file."""
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.vision_projector.state_dict(), f"{save_path}/vision.bin")

    def update_vision_layer(self, load_path):
        """Load and update the vision projector's parameters from the specified file."""
        state_dict = torch.load(f"{load_path}/vision.bin", map_location="cpu", weights_only=True)
        self.vision_projector.load_state_dict(state_dict)

    def save_audio_layer(self, save_path):
        """Save the audio projector's parameters to the specified file."""
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.audio_projector.state_dict(), f"{save_path}/audio.bin")

    def update_audio_layer(self, load_path):
        """Load and update the audio projector's parameters from the specified file."""
        state_dict = torch.load(f"{load_path}/audio.bin", map_location="cpu", weights_only=True)
        self.audio_projector.load_state_dict(state_dict)

    def save_lora_parameters(self, save_path):
        """Save all parameters starting with 'lora_' from the LLaMA model."""
        os.makedirs(save_path, exist_ok=True)
        self.llama.save_pretrained(save_path)

    def freeze_all(self):
        """Freeze all parameters of the model."""
        self.freeze_llama()
        self.freeze_audio_projection_layers()
        self.freeze_vision_projection_layers()

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

    def unfreeze_vision_projection_layers(self):
        """Unfreeze vision projection layers."""
        for param in self.vision_projector.parameters():
            param.requires_grad = True

    def unfreeze_audio_projection_layers(self):
        """Unfreeze audio projection layers."""
        for param in self.audio_projector.parameters():
            param.requires_grad = True

    def freeze_vision_projection_layers(self):
        """Freeze vision projection layers."""
        for param in self.vision_projector.parameters():
            param.requires_grad = False

    def freeze_audio_projection_layers(self):
        """Freeze audio projection layers."""
        for param in self.audio_projector.parameters():
            param.requires_grad = False

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
                
                indices_vid_start = (input_id == DEFAULT_IM_START_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_vid_start = indices_vid_start[0].item() if indices_vid_start.numel() > 0 else None
                indices_vid_end = (input_id == DEFAULT_IM_END_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_vid_end = indices_vid_end[0].item() if indices_vid_end.numel() > 0 else None

                seq_len = int(input_id.ne(self.tokenizer.pad_token_id).sum())
                target_masked_len = (label == IGNORE_TOKEN_ID).sum()
                valid_label = label[target_masked_len:seq_len]
                if False:  # Inspect and check the correctness of masking
                    z = valid_label.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                    print(self.tokenizer.decode(z))
                    exit()
                prefix_id = input_id[:first_vid_start+1]
                suffix_id = input_id[first_vid_end:seq_len]
                prefix_embeds = self.llama.get_input_embeddings()(prefix_id)  
                suffix_embeds = self.llama.get_input_embeddings()(suffix_id)  
                # -1 for we remove the image token place holder
                new_target_mask_len = target_masked_len - 1 + visual_embed.shape[0]
                input_embds = torch.cat([prefix_embeds, visual_embed, suffix_embeds], dim=0)
                pad_embds = torch.zeros((self.tokenizer.model_max_length - input_embds.shape[0], input_embds.shape[-1])).cuda().to(torch.float16)
                input_embds = torch.cat([input_embds, pad_embds], dim=0)
                pad_length = self.tokenizer.model_max_length - new_target_mask_len - len(valid_label)
                new_label = torch.cat([torch.full((new_target_mask_len,), IGNORE_TOKEN_ID).cuda(), valid_label, torch.full((pad_length,), self.tokenizer.pad_token_id).cuda()], dim=0)
                if False:  # Inspect and check the correctness of masking
                    z = new_label.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                    print(self.tokenizer.decode(z))
                    exit()
                
                embeddings.append(input_embds)
                new_labels.append(new_label)
        
        elif images is None:
            
            audio_embeds = self.audio_projector(audio)
            audio_embeds = audio_embeds.view(audio_embeds.shape[0], -1, audio_embeds.shape[-1])

            for input_id, label, audio_embed in zip(input_ids, labels, audio_embeds):
                
                indices_aud_start = (input_id == DEFAULT_AUDIO_START_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_aud_start = indices_aud_start[0].item() if indices_aud_start.numel() > 0 else None
                indices_aud_end = (input_id == DEFAULT_AUDIO_END_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_aud_end = indices_aud_end[0].item() if indices_aud_end.numel() > 0 else None

                seq_len = int(input_id.ne(self.tokenizer.pad_token_id).sum())
                target_masked_len = (label == IGNORE_TOKEN_ID).sum()
                valid_label = label[target_masked_len:seq_len]
                if False:  # Inspect and check the correctness of masking
                    z = valid_label.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                    print(self.tokenizer.decode(z))
                    exit()
                prefix_id = input_id[:first_aud_start+1]
                suffix_id = input_id[first_aud_end:seq_len]
                prefix_embeds = self.llama.get_input_embeddings()(prefix_id)  
                suffix_embeds = self.llama.get_input_embeddings()(suffix_id)  
                # -1 for we remove the image token place holder
                new_target_mask_len = target_masked_len - 1 + audio_embed.shape[0]
                input_embds = torch.cat([prefix_embeds, audio_embed, suffix_embeds], dim=0)
                pad_embds = torch.zeros((self.tokenizer.model_max_length - input_embds.shape[0], input_embds.shape[-1])).cuda().to(torch.float16)
                input_embds = torch.cat([input_embds, pad_embds], dim=0)
                pad_length = self.tokenizer.model_max_length - new_target_mask_len - len(valid_label)
                new_label = torch.cat([torch.full((new_target_mask_len,), IGNORE_TOKEN_ID).cuda(), valid_label, torch.full((pad_length,), self.tokenizer.pad_token_id).cuda()], dim=0)
                if False:  # Inspect and check the correctness of masking
                    z = new_label.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                    print(self.tokenizer.decode(z))
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
                
                # For vision
                indices_vid_start = (input_id == DEFAULT_VID_START_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_vid_start = indices_vid_start[0].item() if indices_vid_start.numel() > 0 else None
                indices_vid_end = (input_id == DEFAULT_VID_END_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_vid_end = indices_vid_end[0].item() if indices_vid_end.numel() > 0 else None
                
                # For Audio
                indices_aud_start = (input_id == DEFAULT_AUDIO_START_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_aud_start = indices_aud_start[0].item() if indices_aud_start.numel() > 0 else None
                indices_aud_end = (input_id == DEFAULT_AUDIO_END_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_aud_end = indices_aud_end[0].item() if indices_aud_end.numel() > 0 else None

                seq_len = int(input_id.ne(self.tokenizer.pad_token_id).sum())
                target_masked_len = (label == IGNORE_TOKEN_ID).sum()
                valid_label = label[target_masked_len:seq_len]
                if False:  # Inspect and check the correctness of masking
                    z = valid_label.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                    print(self.tokenizer.decode(z))
                    exit()
                prefix_id = input_id[:first_vid_start+1]
                midfix_id = input_id[first_vid_end:first_aud_start+1]
                suffix_id = input_id[first_aud_end:seq_len]
                prefix_embeds = self.llama.get_input_embeddings()(prefix_id)  
                midfix_embeds = self.llama.get_input_embeddings()(midfix_id)  
                suffix_embeds = self.llama.get_input_embeddings()(suffix_id)  
                # -2 for we remove the image and audio token place holder
                new_target_mask_len = target_masked_len - 2 + audio_embed.shape[0] + visual_embed.shape[0]
                input_embds = torch.cat([prefix_embeds, visual_embed, midfix_embeds, audio_embed, suffix_embeds], dim=0)
                pad_embds = torch.zeros((self.tokenizer.model_max_length - input_embds.shape[0], input_embds.shape[-1])).cuda().to(torch.float16)
                input_embds = torch.cat([input_embds, pad_embds], dim=0)
                pad_length = self.tokenizer.model_max_length - new_target_mask_len - len(valid_label)
                new_label = torch.cat([torch.full((new_target_mask_len,), IGNORE_TOKEN_ID).cuda(), valid_label, torch.full((pad_length,), self.tokenizer.pad_token_id).cuda()], dim=0)
                if False:  # Inspect and check the correctness of masking
                    z = new_label.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                    print(self.tokenizer.decode(z))
                    exit()
                
                embeddings.append(input_embds)
                new_labels.append(new_label)

        embeddings = torch.stack(embeddings, dim=0)
        new_labels = torch.stack(new_labels, dim=0)
        new_attention_mask = (new_labels != self.tokenizer.pad_token_id).clone().detach().cuda().to(torch.int64)
        new_labels[new_labels == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
               
        outputs = self.llama(
            inputs_embeds=embeddings,
            attention_mask=new_attention_mask,
            labels=new_labels,
        )

        return outputs
    
    def generate(self, input_ids, images=None, audio=None, **generate_kwargs):
   
        embeddings = []
        
        if audio is None:
            is_video = False
            if input_ids.shape[0] != images.shape[0]:
                is_video = True

            image_features = self.vision_tower(images)
            visual_embeds = self.vision_projector(image_features)
            
            if is_video:
                visual_embeds = visual_embeds.view(input_ids.shape[0], -1, *visual_embeds.shape[1:])

            for input_id, visual_embed in zip(input_ids, visual_embeds):
                visual_embed = visual_embed.view(-1, visual_embed.shape[-1])
                
                indices_vid_start = (input_id == DEFAULT_IM_START_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_vid_start = indices_vid_start[0].item() if indices_vid_start.numel() > 0 else None
                indices_vid_end = (input_id == DEFAULT_IM_END_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_vid_end = indices_vid_end[0].item() if indices_vid_end.numel() > 0 else None
                
                prefix_id = input_id[:first_vid_start+1]
                suffix_id = input_id[first_vid_end:]
                
                prefix_embeds = self.llama.get_input_embeddings()(prefix_id)
                suffix_embeds = self.llama.get_input_embeddings()(suffix_id)
                
                input_embds = torch.cat([prefix_embeds, visual_embed, suffix_embeds], dim=0)            
                embeddings.append(input_embds)
        
        elif images is None:
            audio_embeds = self.audio_projector(audio)
            audio_embeds = audio_embeds.view(audio_embeds.shape[0], -1, audio_embeds.shape[-1])
            
            for input_id, audio_embed in zip(input_ids, audio_embeds):
                indices_aud_start = (input_id == DEFAULT_AUDIO_START_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_aud_start = indices_aud_start[0].item() if indices_aud_start.numel() > 0 else None
                indices_aud_end = (input_id == DEFAULT_AUDIO_END_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_aud_end = indices_aud_end[0].item() if indices_aud_end.numel() > 0 else None
                
                prefix_id = input_id[:first_aud_start+1]
                suffix_id = input_id[first_aud_end:]
                
                prefix_embeds = self.llama.get_input_embeddings()(prefix_id)
                suffix_embeds = self.llama.get_input_embeddings()(suffix_id)
                
                input_embds = torch.cat([prefix_embeds, audio_embed, suffix_embeds], dim=0)  
                embeddings.append(input_embds)
        
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
            
            for input_id, visual_embed, audio_embed in zip(input_ids, visual_embeds, audio_embeds):
                visual_embed = visual_embed.view(-1, visual_embed.shape[-1])
                
                indices_vid_start = (input_id == DEFAULT_VID_START_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_vid_start = indices_vid_start[0].item() if indices_vid_start.numel() > 0 else None
                indices_vid_end = (input_id == DEFAULT_VID_END_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_vid_end = indices_vid_end[0].item() if indices_vid_end.numel() > 0 else None
                
                indices_aud_start = (input_id == DEFAULT_AUDIO_START_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_aud_start = indices_aud_start[0].item() if indices_aud_start.numel() > 0 else None
                indices_aud_end = (input_id == DEFAULT_AUDIO_END_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                first_aud_end = indices_aud_end[0].item() if indices_aud_end.numel() > 0 else None
                
                prefix_id = input_id[:first_vid_start+1]
                midfix_id = input_id[first_vid_end:first_aud_start+1]
                suffix_id = input_id[first_aud_end:]
                
                prefix_embeds = self.llama.get_input_embeddings()(prefix_id)
                midfix_embeds = self.llama.get_input_embeddings()(midfix_id)
                suffix_embeds = self.llama.get_input_embeddings()(suffix_id)
    
                input_embds = torch.cat([prefix_embeds, visual_embed, midfix_embeds, audio_embed, suffix_embeds], dim=0)           
                embeddings.append(input_embds)
        
        embeddings = torch.stack(embeddings, dim=0)
        
        outputs = self.llama.generate(
            inputs_embeds=embeddings, max_new_tokens=200
        )
        
        return outputs


    @property
    def config(self):
        return self.llama.config

if __name__ == "__main__":

    parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # TODO: Implement fp8 to enable longer length and larger batch size
    tokenizer.model_max_length = training_args.max_token_length
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN], special_tokens=True)
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
    save_dir = "test"
    model = MultiModalLlama(llama_model_name=llama_model_name, vision_tower=clip_vision_tower, tokenizer=tokenizer)
    model.llama.resize_token_embeddings(len(tokenizer))
    model.llama.config.vocab_size = len(tokenizer)
    id = tokenizer(DEFAULT_VID_START_TOKEN).input_ids
    prefix_embeds = model.llama.get_input_embeddings()(torch.tensor(id))  

    if training_args.stage == "stage_1":

        # Stage 1 of training
        model.freeze_all()               
        model.unfreeze_vision_projection_layers()    
        training_args_stage1 = TrainingArguments(
            output_dir=training_args.model_save_path,
            per_device_train_batch_size=training_args.batch_size,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            logging_steps=training_args.logging_steps,
            fp16=training_args.fp16,
            save_safetensors=False,
            remove_unused_columns=False,  
            deepspeed=training_args.deepspeed_config,
            save_strategy="steps",   
            save_steps=10000,          
            save_total_limit=1  
        )
        trainer_stage1 = Trainer(
            model=model,
            args=training_args_stage1,
            train_dataset=dataset,
            data_collator=collector
        )

        print("=== Stage 1: Training Vision Projection Layers Only ===")
        trainer_stage1.train()
        model.save_vision_layer(training_args.model_save_path)

    elif training_args.stage == "stage_2":

        # Stage 2 of training
        model.freeze_all()  
        model.unfreeze_audio_projection_layers()
        training_args_stage2 = TrainingArguments(
            output_dir=training_args.model_save_path,
            per_device_train_batch_size=training_args.batch_size,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            logging_steps=training_args.logging_steps,
            fp16=training_args.fp16,
            save_safetensors=False,
            remove_unused_columns=False,  
            deepspeed=training_args.deepspeed_config,
            save_strategy="steps",   
            save_steps=10000,          
            save_total_limit=1 
        )
        trainer_stage2 = Trainer(
            model=model,
            args=training_args_stage2,
            train_dataset=dataset,
            data_collator=collector
        )

        print("=== Stage 2: Training Audio Projection Layers Only ===")
        trainer_stage2.train()
        model.save_audio_layer(training_args.model_save_path)

    elif training_args.stage == "stage_3":

        # Stage 3 of training
        model.update_vision_layer(model_args.pretrain_path)
        model.update_audio_layer(model_args.pretrain_path)
        training_args_stage3 = TrainingArguments(
            output_dir=training_args.model_save_path,
            per_device_train_batch_size=training_args.batch_size,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            logging_steps=training_args.logging_steps,
            fp16=training_args.fp16,
            save_safetensors=False,
            remove_unused_columns=False,  
            deepspeed=training_args.deepspeed_config,
            save_strategy="steps",   
            save_steps=10000,          
            save_total_limit=1 
        )
        trainer_stage3 = Trainer(
            model=model,
            args=training_args_stage3,
            train_dataset=dataset,
            data_collator=collector
        )

        print("=== Stage 3: Fine-Tuning LLaMA Only (Projection Layers Frozen) ===")
        trainer_stage3.train()
        model.save_lora_parameters(training_args.model_save_path)
        model.save_audio_layer(training_args.model_save_path)
        model.save_vision_layer(training_args.model_save_path)
        