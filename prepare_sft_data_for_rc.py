import json

# 你的原始数据
data = [...]  # 你的 JSON 数组
# function_call_train_data.json
# 转换为 SFT 格式
sft_data = []
prompt = data[0]["content"]
response = ""
for item in data[1:]:
    if item["type"] == "function_call":
        response += json.dumps({"name": item["name"], "arguments": json.loads(item["arguments"])}) + "\n"
    elif item["type"] == "function_call_output":
        response += f"[TOOL_OUTPUT]{item['output']}\n"
    elif item["type"] == "message" and item["role"] == "assistant":
        response += item["content"][0]["text"] + "<|eos|>"

sft_entry = {
    "prompt": f"<|user|>{prompt}<|assistant|>{response}"
}
sft_data.append(sft_entry)

# 保存为 JSONL
with open("~/data/sft_fault_data.jsonl", "w", encoding="utf-8") as f:
    for item in sft_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print("SFT data saved to ~/data/sft_fault_data.jsonl")