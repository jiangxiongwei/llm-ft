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
lora_model_path = "lora_model"  # 微调后的 LoRA 模型（替换为实际路径）

# 加载模型
try:
    # 加载基础模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=1024,
        dtype=torch.float16,
        load_in_4bit=False,
        device_map={"": 0},
        trust_remote_code=True
    )
    print("Base model loaded successfully")
    
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(
        model,
        lora_model_path,
        device_map={"": 0},
        is_trainable=False  # 推理模式
    )
    print("LoRA adapter loaded successfully")
    
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
    "你喜欢购物吗？",
    "我想看看有哪些脚本",
    "我想看看有哪些计划",
    "北京气温多少",
    "上海天气预报",
    "广州天气怎么样"  # 扩展用例
]

# 验证
print("\n=== Validation Results ===")
# for prompt in test_prompts:
#     try:
#         # 构造输入
#         input_text = f"<|user|>\n{prompt}\n<|assistant|>"
#         inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")
        
#         # 生成输出
#         streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=128,
#             streamer=streamer,
#             pad_token_id=tokenizer.eos_token_id,
#         )
        
#         # 解码输出
#         decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
#         print(f"Input: {prompt}")
#         print(f"Output: {decoded_output}\n")
#     except Exception as e:
#         print(f"Error processing prompt '{prompt}': {e}")


for prompt in test_prompts:
        input_text = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            # streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Input: {prompt}")
        print(f"Output: {decoded_output}\n")
