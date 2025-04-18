import torch
import os
from unsloth import FastLanguageModel
from trl import SFTTrainer, DPOTrainer
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainingArguments
from datasets import load_dataset
import json
from transformers import TextStreamer

# 设置单张 RTX 3090（cuda:0）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# 本地模型路径（替换为你的实际路径）
local_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"

# 数据集路径
data_path = "function_call_train_data_qwen.jsonl"

# 加载数据集
try:
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Dataset loaded successfully: {len(dataset)} samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# 拆分训练和验证集（可选，10% 验证）
if len(dataset) >= 10000:  # 确保有足够数据拆分
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
else:
    train_dataset = dataset
    eval_dataset = None
    print("Dataset too small, skipping validation split")

# 加载预训练模型
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_model_path,
        max_seq_length=1024,
        dtype=torch.float16,
        load_in_4bit=False, 
        device_map={"": 0},
        trust_remote_code=True
    )
    print("Model loaded from:", local_model_path)
    # 检查设备
    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device: {param.device}")
        break
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 配置 LoRA
try:
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        use_gradient_checkpointing=True  # 节省显存
    )
    print("LoRA configured successfully")
except Exception as e:
    print(f"Error configuring LoRA: {e}")
    exit(1)

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
        max_steps=225 if len(train_dataset) <= 600 else 450,  # ~3 epochs
        learning_rate=1e-4,
        fp16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="adamw_torch",
        report_to="none",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
      #  evaluation_strategy="steps" if eval_dataset else "no",
      eval_strategy="no",
        eval_steps=50 if eval_dataset else None,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True if eval_dataset else False,
    ),
)

# 开始训练
try:
    trainer.train()
    print("Training completed")
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# 保存模型
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("Model saved to: lora_model")

# 验证效果
try:
    FastLanguageModel.for_inference(model)
    test_prompts = [
        "杭州天气如何",
        "你喜欢狗吗？",
        "我想查看产品列表",
        "北京气温多少"
    ]
    print("\n=== Validation Results ===")
    for prompt in test_prompts:
        input_text = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Input: {prompt}")
        print(f"Output: {decoded_output}\n")
except Exception as e:
    print(f"Error during validation: {e}")
