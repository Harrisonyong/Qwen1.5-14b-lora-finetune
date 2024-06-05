#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import torch
import transformers
from typing import List, Dict
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from jinja2 import FileSystemLoader, Environment
from config import Config

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class DataSplit:
    """
        divide dataset to train and eval
    """
    data_path = Config.data_path
    val_set_size = Config.val_set_size
    
    @classmethod
    def load_data(cls):
        with open(cls.data_path, "r") as f:
            data = json.load(f)
        return data
    
    @classmethod
    def gen_data(cls):
        data = cls.load_data()
        random.shuffle(data)

        train_data = data[cls.val_set_size:]
        valid_data = data[:cls.val_set_size]
        
        return train_data, valid_data


class QwenDataset(Dataset):
    """
        dataset for supervised fine_tuning
    """
    def __init__(self, raw_data:List, tokenizer:transformers.PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = Config.sequence_len
        data_dict = self.preprocess(raw_data)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i)-> Dict[str, torch.Tensor]:
        return dict(
            input_ids = self.input_ids[i], 
            labels = self.labels[i],
            attention_mask = self.attention_mask[i]
            )

    def preprocess(self,raw_data:List[dict]):
        """Preprocesses the data for supervised fine-tuning."""
        raw_texts = []
        # 获取jinja的加载目录
        j2_loader = FileSystemLoader(os.path.dirname(__file__))
        # 定义jinja2的搜索环境
        env = Environment(loader=j2_loader)
        # 加载模版文件
        qwen_template = env.get_template("Template.j2")
        
        for single_data in raw_data:
            # 将单条数据dict，映射到chat_template中，组成chat_template格式
            # 单条数据应用到模版中组成一个输入
            template_data = qwen_template.render(messages = single_data)
            # 组装成chat_template之后进行token化
            tokenized_data = self.tokenizer(
                template_data,
                max_length=self.max_len,
                padding="max_length",
                truncation=True
            ).input_ids
            raw_texts.append(tokenized_data)
        
        input_ids = torch.tensor(raw_texts, dtype=torch.int)
        target_ids = input_ids.clone()
        target_ids[target_ids == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(
            input_ids = input_ids,
            labels = target_ids,
            attention_mask = attention_mask
        )
