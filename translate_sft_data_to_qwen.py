import json
import re

# 输入和输出文件路径
input_file = "./function_call_train_data.json"
output_file = "./function_call_train_data_qwen.jsonl"

# 读取原始数据
try:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: File {input_file} not found.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Invalid JSON in {input_file}.")
    exit(1)

# 转换为 Qwen2.5 聊天模板格式
sft_data = []
for item in data:
    prompt = item.get("prompt", "")
    # 提取用户和助理内容
    user_match = re.search(r"<\|user\|>\n(.*?)\n<\|assistant\|>\n(.*?)(<|eos|>|$)", prompt, re.DOTALL)
    if not user_match:
        print(f"Warning: Skipping invalid prompt: {prompt[:50]}...")
        continue
    
    user_content = user_match.group(1).strip()
    assistant_content = user_match.group(2).strip()

    # 构建新 prompt
    new_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    new_prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n"
    new_prompt += "<|im_start|>assistant\n"
    
    # 判断是否为 Function Call
    try:
        json.loads(assistant_content)
        new_prompt += f"<tool_call>\n{assistant_content}\n</tool_call>"
    except json.JSONDecodeError:
        new_prompt += assistant_content
    
    new_prompt += "<|im_end|><|endoftext|>"

    sft_data.append({"prompt": new_prompt})

# 保存为 JSONL
try:
    with open(output_file, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Converted SFT data saved to {output_file}")
except Exception as e:
    print(f"Error writing to {output_file}: {e}")