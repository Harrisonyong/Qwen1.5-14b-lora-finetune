#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from peft import LoraConfig, get_peft_model


class QwenLoraModel:
    @classmethod
    def model(cls, *args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            Config.base_model,
            torch_dtype = torch.float16,
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r = Config.lora_r,
            lora_alpha= Config.lora_alpha,
            lora_dropout=Config.lora_dropout,
            target_modules=Config.lora_target_modules,
            bias="none"
        )
        lora_model = get_peft_model(model, lora_config)
        return lora_model

    
class QwenTokenizer:
    @classmethod
    def tokenize(cls,*args, **kwargs):
        tokenize = AutoTokenizer.from_pretrained(
            Config.base_model,
            padding_side="right",
            model_max_length=Config.sequence_len,
            trust_remote_code=True
            )
        return tokenize