import json
import random
import os

# 读取原始数据
def load_raw_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # 假设每行是一个 JSON 数组（路径）
        data = [json.loads(line) for line in f]
    return data

# 转换为 SFT 格式
def convert_to_sft(dialogs):
    sft_data = []
    for dialog in dialogs:
        # 筛选成功路径
        if dialog[-1].get("result") == "True":
            prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            for item in dialog[:-1]:  # 排除 result
                if item.get("role") == "user":
                    prompt += f"<|im_start|>user\n{item['content']}<|im_end|>\n"
                elif item.get("type") == "function_call":
                    arguments = item["arguments"]
                    try:
                        # 确保 arguments 是有效的 JSON 字符串
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON in arguments: {arguments}")
                        continue
                    prompt += f"<|im_start|>assistant\n<tool_call>\n{json.dumps({'name': item['name'], 'arguments': arguments})}\n</tool_call><|im_end|>\n"
                elif item.get("type") == "function_call_output":
                    prompt += f"<|im_start|>user\n<tool_response>\n{item['output']}\n</tool_response><|im_end|>\n"
                elif item.get("role") == "assistant" and item.get("type") == "message":
                    text = item["content"][0]["text"]
                    prompt += f"<|im_start|>assistant\n{text}<|im_end|>\n"
            prompt += "<|endoftext|>"
            sft_data.append({"prompt": prompt})
    return sft_data

# 转换为 DPO 格式
def convert_to_dpo(dialogs):
    dpo_data = []
    # 分离成功和失败路径
    success_dialogs = [d for d in dialogs if d[-1].get("result") == "True"]
    failure_dialogs = [d for d in dialogs if d[-1].get("result") == "False"]
    
    # 配对（假设成功和失败路径数量相等，或随机配对）
    pairs = min(len(success_dialogs), len(failure_dialogs))
    if pairs == 0:
        print("No valid success-failure pairs found.")
        return dpo_data
    
    for i in range(pairs):
        success = success_dialogs[i]
        failure = failure_dialogs[i]  # 可随机选择：random.choice(failure_dialogs)
        
        # 提取 prompt（用户输入）
        prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        prompt += f"<|im_start|>user\n{success[0]['content']}<|im_end|>\n"
        
        # 提取 chosen（成功路径的助手输出）
        chosen = ""
        for item in success[:-1]:  # 排除 result
            if item.get("type") == "function_call":
                arguments = item["arguments"]
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
                chosen += f"<tool_call>\n{json.dumps({'name': item['name'], 'arguments': arguments})}\n</tool_call>"
            elif item.get("type") == "function_call_output":
                chosen += f"<tool_response>\n{item['output']}\n</tool_response>"
            elif item.get("role") == "assistant" and item.get("type") == "message":
                chosen += item["content"][0]["text"]
        
        # 提取 rejected（失败路径的助手输出）
        rejected = ""
        for item in failure[:-1]:  # 排除 result
            if item.get("type") == "function_call":
                arguments = item["arguments"]
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
                rejected += f"<tool_call>\n{json.dumps({'name': item['name'], 'arguments': arguments})}\n</tool_call>"
            elif item.get("type") == "function_call_output":
                rejected += f"<tool_response>\n{item['output']}\n</tool_response>"
            elif item.get("role") == "assistant" and item.get("type") == "message":
                rejected += item["content"][0]["text"]
        
        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    
    return dpo_data

# 主函数
def main():
    current_directory = os.getcwd()
    print(f"current_directory: {current_directory}")
    raw_data_file = "/home/linux/source/finetuning/root_cause_train/rc_data_with_result.txt"  
    sft_output_file = "/home/linux/source/finetuning/root_cause_train/fault_analysis_sft.jsonl"
    dpo_output_file = "/home/linux/source/finetuning/root_cause_train/fault_analysis_dpo.jsonl"
    
    # 读取数据
    try:
        dialogs = load_raw_data(raw_data_file)
        print(f"Loaded {len(dialogs)} dialogs from {raw_data_file}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 转换为 SFT 数据
    sft_data = convert_to_sft(dialogs)
    with open(sft_output_file, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"SFT data saved to {sft_output_file}, {len(sft_data)} records")
    
    # 转换为 DPO 数据
    dpo_data = convert_to_dpo(dialogs)
    with open(dpo_output_file, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"DPO data saved to {dpo_output_file}, {len(dpo_data)} records")

if __name__ == "__main__":
    main()