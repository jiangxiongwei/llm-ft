import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from trl import DPOTrainer
from datasets import load_dataset
from transformers import TrainingArguments

# 模型和数据路径
base_model_path = "~/models/Qwen2.5-7B-Instruct"
lora_model_path = "~/models/lora_fault_model"
dpo_data_path = "~/data/dpo_fault_data.jsonl"

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_path,
    max_seq_length=1024,
    dtype=torch.float16,
    load_in_4bit=True,
    device_map="mps",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(
    model,
    lora_model_path,
    device_map="mps",
    is_trainable=True
)

# 加载数据
dataset = load_dataset("json", data_files=dpo_data_path)

# 预处理
def preprocess_function(examples):
    prompt_inputs = tokenizer(examples["prompt"], truncation=True, max_length=512)
    chosen_inputs = tokenizer(examples["chosen"], truncation=True, max_length=512)
    rejected_inputs = tokenizer(examples["rejected"], truncation=True, max_length=512)
    return {
        "prompt_input_ids": prompt_inputs["input_ids"],
        "prompt_attention_mask": prompt_inputs["attention_mask"],
        "chosen_input_ids": chosen_inputs["input_ids"],
        "chosen_attention_mask": chosen_inputs["attention_mask"],
        "rejected_input_ids": rejected_inputs["input_ids"],
        "rejected_attention_mask": rejected_inputs["attention_mask"]
    }

dataset = dataset.map(preprocess_function, batched=True, num_proc=1)

# DPO 训练
trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    beta=0.1,
    max_prompt_length=512,
    max_length=1024,
    args=TrainingArguments(
        output_dir="~/models/dpo_fault_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=50,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=10,
        save_steps=25,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none"
    ),
    is_encoder_decoder=False
)

# 训练
try:
    trainer.train()
    trainer.save_model("~/models/dpo_fault_model")
    print("DPO training completed")
except Exception as e:
    print(f"Error during DPO training: {e}")