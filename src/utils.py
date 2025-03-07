import torch
from constant import *


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return input_ids

def tokenizer_audio_token(prompt, tokenizer, audio_token_index=AUDIO_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<audio>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if prompt_chunks and prompt_chunks[0] and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [audio_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            import torch
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return input_ids

import re

def tokenizer_image_audio_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                audio_token_index=AUDIO_TOKEN_INDEX, return_tensors=None):
    parts = re.split(r'(<image>|<audio>)', prompt)
    input_ids = []
    first_text = True  

    for part in parts:
        if part == '':
            continue
        if part == '<image>':
            input_ids.append(image_token_index)
        elif part == '<audio>':
            input_ids.append(audio_token_index)
        else:
            tokens = tokenizer(part).input_ids
            if first_text:
                input_ids.extend(tokens)
                first_text = False
            else:
                if tokens and tokens[0] == tokenizer.bos_token_id:
                    input_ids.extend(tokens[1:])
                else:
                    input_ids.extend(tokens)

    if return_tensors is not None:
        if return_tensors == 'pt':
            import torch
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return input_ids
