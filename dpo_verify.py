import torch
import os
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import TextStreamer

# 设置单张 RTX 3090（cuda:0）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# 模型路径
base_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"
lora_model_path = "dpo_model"  # DPO 微调后的 LoRA 模型

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
    print("Model and LoRA adapter loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 启用推理模式
FastLanguageModel.for_inference(model)

# 测试用例
test_prompts = [
    "我想看看支持的产品列表",  # 应为 Function Call
    "我想查看脚本列表",        # 应为聊天
    "我想查看书籍列表",          # 应为聊天
    "请显示支持的产品列表",      # 应为 Function Call
    "昆山天气如何"              # 其他 Function Call
]

# 验证
print("\n=== Validation Results ===")
for prompt in test_prompts:
    try:
        input_text = f"<|user|>\n{prompt}\n<|assistant|>"
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
        print(f"Error processing prompt '{prompt}': {e}")