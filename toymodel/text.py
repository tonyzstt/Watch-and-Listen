import torch.nn as nn
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

class TextDecoder(nn.Module):
    def __init__(self, model_name="gpt2", embed_dim=768, prefix_len=10):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        config = GPT2Config.from_pretrained(model_name)
        config.pad_token_id = self.tokenizer.pad_token_id
        self.model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        self.model.resize_token_embeddings(len(self.tokenizer)) 

        self.prefix_len = prefix_len
        self.proj = nn.Linear(embed_dim, self.model.config.n_embd * prefix_len)

    def forward(
        self,
        fused_features,  
        input_texts=None
    ):
        
        b = fused_features.shape[0]
        pooled = fused_features.mean(dim=1)  

        prefix_embeds = self.proj(pooled)   
        prefix_embeds = prefix_embeds.view(
            b, self.prefix_len, self.model.config.n_embd
        )  

        if input_texts is not None:

            # Training
            # tokenize the gt text
            tokenized = self.tokenizer(
                list(input_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=50
            )
            input_ids = tokenized.input_ids  
            attention_mask = tokenized.attention_mask  
            input_ids = input_ids.to(prefix_embeds.device)
            attention_mask = attention_mask.to(prefix_embeds.device)

            # get the GPT2 token embeddings
            token_embeds = self.model.transformer.wte(input_ids)

            # concat prefix_embeds and token_embeds 
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

            # concat attention masks
            prefix_mask = torch.ones(b, self.prefix_len, device=inputs_embeds.device)
            combined_attention_mask = torch.cat(
                [prefix_mask, attention_mask], dim=1
            )  

            # should ignore prefix
            dummy_prefix_labels = torch.full(
                (b, self.prefix_len), -100, dtype=torch.long, device=inputs_embeds.device
            )
            all_labels = torch.cat([dummy_prefix_labels, input_ids], dim=1)

            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_attention_mask,
                labels=all_labels,
            )

            loss = outputs.loss

            return loss, None

        else:

            # Inference
            generated_texts = []
            for i in range(b):
                single_prefix = prefix_embeds[i].unsqueeze(0)  
                output_ids = self.model.generate(
                    inputs_embeds=single_prefix,
                    max_length=50,
                    num_return_sequences=1,
                    do_sample=False
                )
                text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                generated_texts.append(text)

            return None, generated_texts
