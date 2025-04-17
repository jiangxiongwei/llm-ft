import torch
import os
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import json

# 设置单张 RTX 3090（cuda:0）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# 本地模型路径（替换为你的实际路径）
local_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"

# Alpaca 格式数据（示例，适配你的任务）
alpaca_data = [
    {
        "instruction": "Convert the user query into a JSON function call.",
        "input": "苏州气温",
        "output": "{\"name\": \"get_weather_by_location\", \"arguments\": {\"location\": \"苏州\"}}"
    },
    {
        "instruction": "Analyze the error and provide troubleshooting steps.",
        "input": "错误 503",
        "output": "1. 检查网络\n2. 查看日志"
    },
    {
        "instruction": "Answer the question based on knowledge base.",
        "input": "产品 X 功能？",
        "output": "X 提供分析、报告、云同步。"
    }
    # 600 或 2000 条数据可扩展
]

# 保存 Alpaca 数据
with open("alpaca_data.json", "w", encoding="utf-8") as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

# 加载数据集
dataset = load_dataset("json", data_files="alpaca_data.json", split="train")

# 格式化提示
def formatting_prompts_func(examples):
    texts = [
        f"<|user|>\n{inst}\nInput: {inp}\n<|assistant|>\n{out}<|eos|>"
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# 加载模型
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_model_path,
        max_seq_length=1024,
        dtype=torch.float16,
      #  load_in_4bit=False,
      #  device_map="auto",
        load_in_4bit=False,  # 4-bit 推荐，单卡安全
        device_map={"": 0},  # 强制 cuda:0
        trust_remote_code=True
    )
    print("Model loaded from:", local_model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 配置 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.0,
)

# 训练配置
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        max_steps=225 if len(dataset) <= 600 else 750,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="adamw_torch",
    ),
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained("lora_model")
