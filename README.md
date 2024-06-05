### 前言

本文主要介绍使用transformers和peft针对Qwen1.5-14B大模型进行lora微调，分布式训练采用accelerate+deespeed-zero3

**运行环境为:**

**硬件：**

1. 操作系统：ubuntu22.04

2. GPU:v100(4*32G) 
3. CPU内存380G

**软件：**

1. Nvidia驱动：525.105.17

2. Cuda12.0
3. python3.11.9
4. torch2.3.0
   torchaudio2.3.0
   torchvision 0.18.0
5. accelerate0.30.1
6. deepspeed0.14.0  这里有个版本的坑，后面再说
7. transformers4.40.2

### 数据处理

#### 构造数据

精调数据为QA对的形式，需要组合数据为json文件，其中是一个包含多个QA的list，每个QA对包含对应的query和answer，当然也可以带有instruction作为提示。

｜ 注意这里和官方的数据构造不太一样，所以在下面构造chattemplate的时候也需要改一下数据构造方法

file：data.json

```json
[
  {
    "input":"who are you",
    "output":"I'm 888888"
    }，
  {
  "instruction":"you are a helpful assistant"
  "input":"who are you",
  "output":"I'm 888888"
  },
]
```

#### 组合chattemplate作为模型输入

chattemplate的主要作用是在对话中设置角色、提示、开头和结尾词等，保证聊天效率和一致性

- 下面是针对千问1.5的chattemplate，在上述数据格式下的模版构造方法

File:Template.j2

```jinja2
<|im_start|>system
{%- if messages["instruction"] %}
{{ messages["instruction"] }}<|im_end|>
{%- else %}
You are a helpful assistant.<|im_end|>
{%- endif %}
<|im_start|>user
{{messages['input']}}<|im_end|>
<|im_start|>assistant
{{messages['output']}}<|im_end|>
```

- 构造chat模版后，需要将每一个QA放入到模版中构造input，然后对chat-input进行token化，获得input_ids，target_ids和attention_mask

File：data_helper.py

```python
from jinja2 import FileSystemLoader, Environment

raw_data = [
  {
    "input":"who are you",
    "output":"I'm 888888"
    }，
  {
  "instruction":"you are a helpful assistant"
  "input":"who are you",
  "output":"I'm 888888"
  },
]
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

```

- 组成的raw_text进行token化，作为input_ids，ground_truth=input_ids，attention_mask，将其作为模型输入

```python
input_ids = torch.tensor(raw_texts, dtype=torch.int)
target_ids = input_ids.clone()
target_ids[target_ids == self.tokenizer.pad_token_id] = -100
attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
```

### 加载模型配置lora参数

使用peft-和transformers加载基础模型和peft配置参数，config文件可详见git仓库；

File：model.py

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from peft import LoraConfig, get_peft_model

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
tokenize = AutoTokenizer.from_pretrained(
            Config.base_model,
            padding_side="right",
            model_max_length=Config.sequence_len,
            trust_remote_code=True
            )
lora_model = get_peft_model(model, lora_config)
```

### 训练代码

Metrics评估：通过token的完全匹配，计算evaldataset的精度，因为序列是token by token的方式生成的，因此pred的第一个其实是真实值的第二个，需要注意。

File:mertrics.py

```python
def accuracy(pred):
    """
    计算生成的序列和真实标签的精确度，tokenbytoken的计算是否完全匹配
    
    :param desc: pred trainer_utils.EvalPrediction
    :return: dict accuracy
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    total = 0
    correct = 0
    for pred_y, true_y in zip(preds, labels):
        pred_y = pred_y[:-1]
        true_y = true_y[1:]
        
        for p, t in zip(pred_y, true_y):
            if t != -100:
                total += 1
                if p == t:
                    correct += 1
    return {'accuracy':correct /total if total >0 else 0}
```

使用transformers的Trainer，组装dataset、model、tokenizer、deepspeed策略作为参数传入Trainer中进行训练

File:train.py

```python
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
trainer = Trainer(
            model=self.model, tokenizer=self.tokenizer, args=train_args, **data_module,compute_metrics=accuracy
        )
if list(pathlib.Path(train_args.output_dir).glob("checkpoint-*")):
  trainer.train(resume_from_checkpoint=True)
else:
  trainer.train()
```

### 分布式训练策略

这里采用Deepspeed zero3-offload策略，因为14B模型的本身大小为27GB，加上优化器和梯度等需要的显存大概为360GB左右，4卡无法完全加载进行训练

#### Deepspeed-zero3-offload配置

File:deepspeed_config.json

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "allgather_bucket_size": 1e6,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 1e6,
        "stage3_param_persistence_threshold": 1e6,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "steps_per_print": 1000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

#### accelerate配置

Accelerate的配置使用shell命令进行选择就可以了，不需要手动去创建这个yaml文件，在终端中输入下面的命令

```shell
accelerate config --config_file 你希望防止配置的地址
```

｜注意选择的时候是单个node，多个process也就是多个GPU，命令行结束后就会生成yaml文件

File：deepspeed_acc.yaml

```yaml
compute_environment: LOCAL_MACHINE
debug: true
deepspeed_config:
  deepspeed_config_file: deepspeed_config.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

使用accelerate启动训练

```shell
accelerate launch --config_file deepspeed_acc.yaml train.py
```

### 模型验证

使用generate方法验证输入输出

```python
tokenizer = AutoTokenizer.from_pretrained(Config.base_model)
model = AutoPeftModelForCausalLM.from_pretrained(
            lora_dir,
            torch_dtype = torch.float16,
            device_map = "auto",
            trust_remote_code=True)
with torch.no_grad():
    generated_ids = model.generate(
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
```

### 模型保存

上述训练过程中，会保存lora训练完成后的参数，这里需要将参数合并到基础模型上，方便transformers直接使用，直接使用transformers.AutoModelForCausalLM加载模型

```python
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
```

### 遇到的坑

1. 安装包的坑：deepspeed包0.14.2存在offload损失到不同设备的问题，在计算反向的时候，梯度数据卸载到cpu，导致计算设备报错，将版本回退到0.14.0可以解决。报错如下：

   <img width="1392" alt="image" src="https://github.com/Harrisonyong/Qwen1.5-14b-lora-finetune/assets/31169791/2e7fe383-0e96-45cc-9660-f265ebe43151">

