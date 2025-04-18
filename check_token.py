from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/linux/llm/Qwen2.5-7B-Instruct", trust_remote_code=True)
print(tokenizer.special_tokens_map)
print(tokenizer.encode("<|eos|>")) #多个token
print(tokenizer.encode("<|endoftext|>")) # 单一 token
print(tokenizer.encode("<|im_start|>"))  # 单一 token
print(tokenizer.encode("<|im_end|>")) # 单一 token

print(tokenizer.chat_template)  # 确认默认模板