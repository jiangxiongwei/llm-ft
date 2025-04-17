import torch
import os
import json
import random
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import TextStreamer

# 设置单张 RTX 3090（cuda:0）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# 模型路径
base_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"
lora_model_path = "lora_model"  # SFT 微调后的 LoRA
sft_data_path = "function_call_train_data.json"  # SFT 数据
dpo_data_path = "dpo_data.json"  # 输出 DPO 数据

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
        is_trainable=False
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 加载 SFT 数据
try:
    with open(sft_data_path, "r", encoding="utf-8") as f:
        sft_data = json.load(f)
    print("SFT data loaded successfully")
except Exception as e:
    print(f"Error loading SFT data: {e}")
    exit(1)

# 测试输入（生成幻觉）
test_inputs = [
    "我想查看脚本列表",
    "我想查看书籍列表",
    "我想查看电影列表",
    "我想查看音乐列表"
]

# 生成模型输出
def generate_output(prompt):
    try:
        input_text = f"<|user|>\n{prompt}\n<|assistant|>"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
        assistant_text = decoded.split("<|assistant|>")[1].split("<|eos|>")[0]
        return assistant_text.strip()
    except Exception as e:
        print(f"Error generating for '{prompt}': {e}")
        return ""

# 敷衍回答列表
chat_rejected = ["好的。", "嗯。", "还行。", "不知道。"]

# DPO 数据
dpo_data = []

# 方法 1：生成幻觉负样本
for prompt in test_inputs:
    output = generate_output(prompt)
    list_type = prompt.split("查看")[1].replace("列表", "").strip()
    dpo_entry = {
        "messages": [{"role": "user", "content": prompt}],
        "chosen": f"好的，你想查看什么类型的{list_type}列表？",
        "rejected": output
    }
    dpo_data.append(dpo_entry)

# 方法 2：从 SFT 数据派生负样本
for item in sft_data:
    prompt = item["prompt"].split("<|user|>")[1].split("<|assistant|>")[0].strip()
    response = item["prompt"].split("<|assistant|>")[1].split("<|eos|>")[0].strip()
    
    # Function Call 样本
    if "get_xsea_product_list" in response:
        dpo_entry = {
            "messages": [{"role": "user", "content": prompt}],
            "chosen": response,
            "rejected": random.choice([
                "好的，我帮你查产品列表。",
                "{\"name\": \"get_xsea_wrong_list\", \"arguments\": {}}"
            ])
        }
    # 聊天样本
    else:
        dpo_entry = {
            "messages": [{"role": "user", "content": prompt}],
            "chosen": response,
            "rejected": random.choice(chat_rejected)
        }
    dpo_data.append(dpo_entry)

# 保存 DPO 数据
try:
    with open(dpo_data_path, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    print(f"DPO data saved to {dpo_data_path}")
except Exception as e:
    print(f"Error saving DPO data: {e}")