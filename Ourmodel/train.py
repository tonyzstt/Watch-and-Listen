import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from huggingface_hub import login
from vision.vision_projection import build_vision_projector
from audio.audio_projection import build_audio_projector
from config.config import VisionProjectorConfig, AudioProjectorConfig
from peft import LoraConfig, get_peft_model
from audio.wav2vec_encoder import AudioEncoder
from vision.clip_encoder import CLIPVisionTower

def print_gpu_memory_usage(device=0):
    allocated = torch.cuda.memory_allocated(device) / 1024**2  
    reserved = torch.cuda.memory_reserved(device) / 1024**2   
    print(f"GPU {device} - Allocated Memory: {allocated:.2f} MB")
    print(f"GPU {device} - Reserved Memory: {reserved:.2f} MB")

class MultiModalDataset(Dataset):
    """
    Each example should contain:
      - "visual_tokens": a list or tensor of visual features.
      - "audio_tokens": a list or tensor of audio features.
      - "question": the text question.
      - "answer": the ground truth answer.
    
    The final text prompt is created as:
         "Q: {question} A: {answer}"
    with the prompt portion masked in the labels.
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = f"Q: {item['question']} A:"
        full_text = prompt + " " + item["answer"]

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)

        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_ids = prompt_encoding["input_ids"].squeeze(0)
        prompt_len = (prompt_ids != self.tokenizer.pad_token_id).sum().item()
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        visual_tokens = torch.tensor(item["visual_tokens"], dtype=torch.float)
        audio_tokens = torch.tensor(item["audio_tokens"], dtype=torch.float)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "visual_tokens": visual_tokens,
            "audio_tokens": audio_tokens,
        }

class MultiModalLlama(nn.Module):
    def __init__(
        self,
        llama_model_name="meta-llama/Llama-3.2-3B",
        vision_config_params=None,
        audio_config_params=None,
        use_lora=True,
    ):
        super(MultiModalLlama, self).__init__()

        self.llama = AutoModelForCausalLM.from_pretrained(llama_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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
                "hidden_size": 3072,
            }
        if audio_config_params is None:
            audio_config_params = {
                "mm_projector_type": "mlp2x_gelu",
                "audio_hidden_size": 768,
                "hidden_size": 3072,
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

    def unfreeze_projection_layers(self):
        """Unfreeze vision and audio projection layers."""
        for param in self.vision_projector.parameters():
            param.requires_grad = True
        for param in self.audio_projector.parameters():
            param.requires_grad = True

    def forward(self, input_ids, visual_tokens, audio_tokens, labels=None):
        """
        Input:
          - input_ids: tokenized text (question+answer) of shape (batch, seq_len)
          - visual_tokens: visual features of shape (batch, v_len, feat_dim)
          - audio_tokens: audio features of shape (batch, a_len, feat_dim)
          - labels: token labels for LM loss.
        """
        text_embeds = self.llama.get_input_embeddings()(input_ids)  

        # Projection
        visual_embeds = self.vision_projector(visual_tokens)   
        audio_embeds = self.audio_projector(audio_tokens)        
        prefix_embeds = torch.cat([visual_embeds, audio_embeds], dim=1)  

        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

        batch_size = input_ids.size(0)
        prefix_len = prefix_embeds.size(1)

        prefix_attention_mask = torch.ones((batch_size, prefix_len), device=input_ids.device)
        text_attention_mask = (input_ids != self.tokenizer.pad_token_id)
        attention_mask = torch.cat([prefix_attention_mask, text_attention_mask], dim=1)

        if labels is not None:
            prefix_labels = torch.full((batch_size, prefix_len), -100, device=labels.device)
            labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        print_gpu_memory_usage()
        return outputs

    @property
    def config(self):
        return self.llama.config

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.llama.prepare_inputs_for_generation(input_ids, **kwargs)

# Test
def main():

    # Dummy Data
    dummy_data = [
        {
            "visual_tokens": [[0.1] * 768] * 5,  
            "audio_tokens": [[0.2] * 768] * 3,    
            "question": "What",
            "answer": "A",
        },
        {
            "visual_tokens": [[0.3] * 768] * 5,
            "audio_tokens": [[0.4] * 768] * 3,
            "question": "What",
            "answer": "The",
        },
    ]

    login(token='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')
    llama_model_name = "meta-llama/Llama-3.2-3B"  
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = MultiModalLlama(llama_model_name=llama_model_name)
    dataset = MultiModalDataset(dummy_data, tokenizer, max_length=256)

    # clip_model_name="openai/clip-vit-base-patch32"
    # audio_model_name="facebook/wav2vec2-base-960h"
    # clip_vision_tower = CLIPVisionTower(clip_model_name)
    # audio_encoder = AudioEncoder(audio_model_name)

    # Stage 1 of training
    model.freeze_llama()                 
    model.unfreeze_projection_layers()    
    training_args_stage1 = TrainingArguments(
        output_dir="./llama_finetuned_stage1",
        per_device_train_batch_size=8,
        num_train_epochs=20,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        save_safetensors=False,
        remove_unused_columns=False,  
        deepspeed="deepspeed.json"
    )
    trainer_stage1 = Trainer(
        model=model,
        args=training_args_stage1,
        train_dataset=dataset,
    )

    print("=== Stage 1: Training Projection Layers Only ===")
    trainer_stage1.train()

    # Stage 2 of training
    model.freeze_projection_layers()  
    model.unfreeze_llama()           
    training_args_stage2 = TrainingArguments(
        output_dir="./llama_finetuned_stage2",
        per_device_train_batch_size=8,
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
    )

    print("=== Stage 2: Fine-Tuning LLaMA Only (Projection Layers Frozen) ===")
    trainer_stage2.train()

if __name__ == "__main__":
    main()
