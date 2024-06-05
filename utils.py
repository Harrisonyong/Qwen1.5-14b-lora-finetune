#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import Config

def merge_and_save_model(lora_dir, output_dir):
    print(f"the lora dir is {lora_dir}, the ouput dir is {output_dir}")
    base_tokenizer = AutoTokenizer.from_pretrained(Config.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    lora_model = PeftModel.from_pretrained(
        base_model, lora_dir, torch_dtype=torch.float16
    )
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    base_tokenizer.save_pretrained(output_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir",type=str,
                        help = "the lora dir have been finetuned")
    parser.add_argument("--merged_dir",type = str,
                        help="the output dir merged by base and finetuned model")
    args = parser.parse_args()
    merge_and_save_model(lora_dir=args.lora_dir, output_dir=args.merged_dir)