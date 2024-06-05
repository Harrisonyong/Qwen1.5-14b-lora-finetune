#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import pathlib
project_path = os.path.abspath(".")
sys.path.insert(0, project_path)
from transformers import TrainingArguments, Trainer
from model import QwenLoraModel, QwenTokenizer
from data_helper import DataSplit, QwenDataset
from config import Config
from metrics import accuracy

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

train_args = TrainingArguments(
    output_dir=Config.output_path,
    num_train_epochs=Config.epochs,
    per_device_train_batch_size=Config.micro_batch_size,
    per_device_eval_batch_size=Config.micro_eval_size,
    gradient_accumulation_steps=Config.accu_steps,
    evaluation_strategy=Config.evaluation_strategy,
    eval_steps=Config.eval_steps,
    logging_steps=Config.log_every,
    save_steps=Config.checkpoint_every,
    save_total_limit=Config.save_total_limit,
    # max_steps=Config.train_steps,
    learning_rate=Config.learning_rate,
    lr_scheduler_type=Config.lr_scheduler_type,
    warmup_steps=Config.warmup_steps,
    weight_decay=Config.weight_decay,
    adam_beta1=Config.adam_beta1,
    adam_beta2=Config.adam_beta2,
    fp16=True,
    load_best_model_at_end=True,
    deepspeed=Config.deepspeed_config,
    report_to="none",
)


class QwenTrainer:
    """
        定义训练类
    """
    def __init__(self):
        self.tokenizer = QwenTokenizer.tokenize()
        print("tokenizer init done: ", len(self.tokenizer))
        
        self.train_data_set, self.valid_data_set = self.get_data_load()
        print("get data loader done")
        
        self.model = QwenLoraModel.model()
        print("model load done")
        
        self.model.print_trainable_parameters()

    
    def get_data_load(self):
        """
            加载数据集为dataloader
        """
        rank0_print("Loading data...")
        train_data, valid_data = DataSplit.gen_data()
        

        train_dataset = QwenDataset(train_data,self.tokenizer)
        valid_dataset = QwenDataset(valid_data, self.tokenizer)
        rank0_print(f"train_data_size:{len(train_dataset)},valid_data_size:{len(valid_dataset)}")
        
        return dict(train_dataset=train_dataset, eval_dataset=valid_dataset)

    def train(self):
        global local_rank
        start = time.time()
        local_rank = train_args.local_rank
        data_module = self.get_data_load()
        trainer = Trainer(
            model=self.model, tokenizer=self.tokenizer, args=train_args, **data_module,compute_metrics=accuracy
        )
        if list(pathlib.Path(train_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()
        final_dir = train_args.output_dir + "/endpoint"
        trainer.save_model(final_dir)
        end = time.time()
        print("total train time: ", end - start)
        print("model save done")
        
def main():
    qwentrainer = QwenTrainer()
    qwentrainer.train()
            

if __name__ == '__main__':
    main()