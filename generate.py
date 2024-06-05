#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

project_path = os.path.abspath(".")
sys.path.insert(0, project_path)

from typing import List, Tuple
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from config import Config


class Generator:
    def __init__(self,lora_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.base_model)
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            lora_dir,
            torch_dtype = torch.float16,
            device_map = "auto",
            trust_remote_code=True,
        )
    
    def make_context(
        self,
        query: str,
        max_window_size: int,
        history: List[Tuple[str, str]] = None,
        system: str = "",
    ):
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [151644]
        im_end_tokens = [151645]
        nl_tokens = self.tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", self.tokenizer.encode(
                role
            ) + nl_tokens + self.tokenizer.encode(content)

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []
        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens
            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )
            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break
        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + self.tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"
        return raw_text, context_tokens
    
    def evaluate(self, 
                 instruction="You are a helpful assistant.", 
                 input=None, 
                 temperature = 0.7,
                 top_p = 0.75,
                 top_k =20,
                 max_new_tokens = 512,
                 ):
        _, context_tokens = self.make_context(
            query=input,
            max_window_size=max_new_tokens, 
            history=[],
            system=instruction
        )
        input_ids = torch.tensor([context_tokens]).to(self.model.device)
    
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids = input_ids,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                top_k = top_k
            )
            generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(response)

