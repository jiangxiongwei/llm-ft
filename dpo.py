import torch
import os
from unsloth import FastLanguageModel
from peft import PeftModel
from trl import DPOTrainer
from datasets import load_dataset
from transformers import TrainingArguments

# 设置单张 RTX 3090（cuda:0）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# 模型路径
base_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"
lora_model_path = "lora_model"
dpo_data_path = "dpo_data.json"

# 加载模型
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=1024,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map={"": 0},
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(
        model,
        lora_model_path,
        device_map={"": 0},
        is_trainable=True
    )
    print("Model and LoRA adapter loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 加载 DPO 数据
try:
    dataset = load_dataset("json", data_files=dpo_data_path)
    print("DPO dataset loaded successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# 预处理数据集
def preprocess_dataset(example):
    if not example.get("chosen") or not example.get("rejected"):
        example["chosen"] = "默认回答"
        example["rejected"] = "默认错误回答"
    if not example.get("messages"):
        example["messages"] = [{"role": "user", "content": "空输入"}]
    return example

dataset = dataset.map(preprocess_dataset, num_proc=1)  # 单线程，避免多进程问题

# DPO 训练
trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    beta=0.1,
    args=TrainingArguments(
        output_dir="dpo_outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=200,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        optim="adamw_torch",
        remove_unused_columns=False  # 保留所有列
    )
)

# 开始训练
try:
    trainer.train()
    print("DPO training completed")
    model.save_pretrained("dpo_model")
except Exception as e:
    print(f"Error during DPO training: {e}")