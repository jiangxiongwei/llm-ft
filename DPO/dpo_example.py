import torch
import unsloth
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel
import os

# 1. 设置环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定 GPU
torch.cuda.empty_cache()  # 清空 GPU 缓存

# 2. 加载本地 Qwen2.5-7B 模型和分词器
model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"  # 替换为本地模型路径
max_seq_length = 2048

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=torch.float16,  # 使用 FP16 降低显存
        load_in_4bit=True,    # 使用 4-bit 量化
        device_map="auto"
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 启用 Unsloth 优化（LoRA 适配器）
try:
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA 秩
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        max_seq_length=max_seq_length
    )
except Exception as e:
    print(f"Error applying LoRA: {e}")
    exit(1)

# 3. 加载 DPO 数据集
try:
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
    print("Dataset loaded successfully.")
    print("Column names:", dataset.column_names)
    print("First sample:", dataset[0])
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# 4. 数据预处理
def format_dpo_data(example):
    try:
        # 合并 system 和 question 为 prompt
        prompt = f"{example['system']}\n{example['question']}" if example['system'] else example['question']
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }
    except KeyError as e:
        print(f"KeyError in format_dpo_data: {e}, example: {example}")
        raise

try:
    dataset = dataset.map(format_dpo_data)
    print("Dataset mapping successful.")
except Exception as e:
    print(f"Error mapping dataset: {e}")
    exit(1)

# 5. 配置 DPO 训练参数
dpo_config = DPOConfig(
    output_dir="./dpo_output_qwen2.5",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    optim="adamw_torch"
)

# 6. 初始化 DPOTrainer
try:
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth 支持无参考模型的 DPO
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=0.1,  # DPO 的 beta 参数，控制偏好强度
    )
except Exception as e:
    print(f"Error initializing DPOTrainer: {e}")
    exit(1)

# 7. 开始训练
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# 8. 保存微调后的模型
try:
    model.save_pretrained("./dpo_finetuned_qwen2.5")
    tokenizer.save_pretrained("./dpo_finetuned_qwen2.5")
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")
    exit(1)

# 9. 推理示例
try:
    FastLanguageModel.for_inference(model)  # 启用推理模式
    inputs = tokenizer("你好，我能帮你什么？", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50)
    print("Inference output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
except Exception as e:
    print(f"Error during inference: {e}")
    exit(1)