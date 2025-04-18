import json

# 你的原始数据
data = [...]  # 你的 JSON 数组

# 转换为 DPO 格式
dpo_data = []
prompt = data[0]["content"]
for item in data[1:]:
    if item["type"] == "function_call":
        dpo_entry = {
            "prompt": prompt,
            "chosen": json.dumps({"name": item["name"], "arguments": json.loads(item["arguments"])}),
            "rejected": random.choice([
                "好的，我帮你检查 CPU 使用情况。",
                "{\"name\": \"wrong_tool\", \"arguments\": {}}"
            ])
        }
        dpo_data.append(dpo_entry)
    elif item["type"] == "function_call_output":
        prompt += f"\n[TOOL_OUTPUT]{item['output']}"

# 保存为 JSONL
with open("~/data/dpo_fault_data.jsonl", "w", encoding="utf-8") as f:
    for item in dpo_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print("DPO data saved to ~/data/dpo_fault_data.jsonl")