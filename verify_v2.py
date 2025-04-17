import torch
import os
from unsloth import FastLanguageModel
from transformers import TextStreamer

# 设置单张 RTX 3090（cuda:0）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# 模型路径
base_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"  # 基础模型
lora_model_path = "lora_model"  # 微调后的 LoRA 模型（替换为实际路径）

# 加载模型
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        peft_model_name=lora_model_path,  # 加载 LoRA 适配器
        max_seq_length=1024,
        dtype=torch.float16,
        load_in_4bit=False,  # 4-bit，与微调一致
        device_map={"": 0},  # 强制 cuda:0
        trust_remote_code=True
    )
    print("Model and LoRA adapter loaded successfully")
    # 检查设备
    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device: {param.device}")
        break
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 启用推理模式
FastLanguageModel.for_inference(model)

# 测试用例
test_prompts = [
    "昆山天气如何",
    "你喜欢小宠物吗？",
    "我想查看产品列表",
    "拉萨气温多少",
    "上海天气预报"  # 扩展用例
]

# 验证
print("\n=== Validation Results ===")
for prompt in test_prompts:
    try:
        # 构造输入
        input_text = f"<|user|>\n{prompt}\n<|assistant|>"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")
        
        # 生成输出
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
            sliding_window=4096  # Qwen2.5 滑动窗口
        )
        
        # 解码输出
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Input: {prompt}")
        print(f"Output: {decoded_output}\n")
    except Exception as e:
        print(f"Error processing prompt '{prompt}': {e}")
