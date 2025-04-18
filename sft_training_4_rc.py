import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments

# 模型和数据路径
base_model_path = "~/models/Qwen2.5-7B-Instruct"
sft_data_path = "~/data/sft_fault_data.jsonl"
output_dir = "~/models/lora_fault_model"

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_path,
    max_seq_length=1024,
    dtype=torch.float16,
    load_in_4bit=True,
    device_map="mps",
    trust_remote_code=True
)

# 加载数据
dataset = load_dataset("json", data_files=sft_data_path)

# SFT 训练
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="prompt",
    max_seq_length=1024,
    args=TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=100,  # 调整为数据量
        learning_rate=2e-5,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none"
    )
)

# 训练
try:
    trainer.train()
    trainer.save_model(output_dir)
    print("SFT training completed")
except Exception as e:
    print(f"Error during SFT training: {e}")