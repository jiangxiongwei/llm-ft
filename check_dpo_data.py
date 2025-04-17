import json

with open("dpo_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i, item in enumerate(data):
    if not item.get("messages") or not isinstance(item["messages"], list):
        print(f"Sample {i}: Invalid messages: {item.get('messages')}")
    if not item.get("chosen") or not isinstance(item["chosen"], str):
        print(f"Sample {i}: Invalid chosen: {item.get('chosen')}")
    if not item.get("rejected") or not isinstance(item["rejected"], str):
        print(f"Sample {i}: Invalid rejected: {item.get('rejected')}")
    if item["chosen"] == "" or item["rejected"] == "":
        print(f"Sample {i}: Empty chosen or rejected: {item}")