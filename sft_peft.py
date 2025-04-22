import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# 设置单张 RTX 3090（cuda:0）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# 本地模型路径
local_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"

# 数据集路径
data_path = "function_call_train_data_qwen.jsonl"

# 加载数据集
dataset = load_dataset("json", data_files=data_path, split="train")
print(f"Dataset loaded successfully: {len(dataset)} samples")

# 拆分训练和验证集
if len(dataset) >= 10000:
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
else:
    train_dataset = dataset
    eval_dataset = None
    print("Dataset too small, skipping validation split")

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    load_in_4bit=False,
    device_map={"": 0},
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
print("Model loaded from:", local_model_path)

# 启用梯度检查点
model.gradient_checkpointing_enable()
print("Gradient checkpointing enabled:", model.config.use_cache)

# 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    layers_to_transform=[0, 1, 2, 3, 4],
    task_type="CAUSAL_LM"
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
print("LoRA configured successfully with layers_to_transform:", lora_config.layers_to_transform)

# 检查 trainable 参数
print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: requires_grad={param.requires_grad}")

# 调试损失
model.train()
sample = train_dataset[0]
inputs = tokenizer(sample["prompt"], return_tensors="pt").to("cuda:0")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
print("Sample loss:", loss)
print("Loss requires grad:", loss.requires_grad)
print("Loss grad_fn:", loss.grad_fn)

# 训练配置
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="prompt",
    max_seq_length=1024,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        max_steps=225 if len(train_dataset) <= 600 else 450,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="adamw_torch",
        report_to="none",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=50,
        gradient_checkpointing=True
    ),
)

# 开始训练
trainer.train()
print("Training completed")

# 保存模型
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("Model saved to: lora_model")