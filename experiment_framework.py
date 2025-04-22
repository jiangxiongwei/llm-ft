import torch
import os
import json
import random
import math
import shutil
import gc
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

local_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"

weather_templates = {
    "low": ["{location}现在气温多少"],
    "mid": ["{location}现在气温多少", "{location}今天天气如何？", "请告诉我{location}的天气"],
    "high": [
        "{location}现在气温多少",
        "{location}今天天气如何？",
        "明天{location}会下雨吗？",
        "请告诉我{location}的天气",
        "{location}的气候怎么样？"
    ]
}
product_templates = {
    "low": ["我想看看产品列表"],
    "mid": ["我想看看产品列表", "列出所有产品", "有哪些产品可用？"],
    "high": [
        "我想看看产品列表",
        "列出所有产品",
        "有哪些产品可用？",
        "给我看产品列表",
        "查看产品目录"
    ]
}
neg_templates = {
    "low": ["你喜欢狗吗？", "今天是星期几？"],
    "mid": ["你喜欢狗吗？", "今天是星期几？", "列出所有脚本"],
    "high": [
        "列出所有脚本",
        "什么是脚本？",
        "你喜欢狗吗？",
        "今天是星期几？",
        "讲个笑话"
    ]
}
locations = {
    "low": ["苏州", "北京"],
    "mid": ["苏州", "北京", "上海", "广州"],
    "high": ["苏州", "北京", "上海", "广州", "New York", "London"]
}

def generate_samples(size, diversity="high", neg_ratio=0.3):
    samples = []
    neg_samples = int(size * neg_ratio)
    pos_samples = size - neg_samples

    for _ in range(pos_samples):
        system_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant capable of executing function calls. Available functions: get_xsea_product_list(), get_weather_by_location(location). Respond with a <tool_call> containing a JSON object with 'name' and 'arguments'. For non-function queries, respond with plain text.<|im_end>"
        if random.random() < 0.5:
            loc = random.choice(locations[diversity])
            prompt = random.choice(weather_templates[diversity]).format(location=loc)
            sample = {
                "prompt": f"{system_prompt}\n<|im_start|>user\n{prompt}<|im_end>",
                "assistant": f"<tool_call>{{\"name\": \"get_weather_by_location\", \"arguments\": {{\"location\": \"{loc}\"}}}}</tool_call><|im_end><|endoftext|>"
            }
        else:
            prompt = random.choice(product_templates[diversity])
            sample = {
                "prompt": f"{system_prompt}\n<|im_start|>user\n{prompt}<|im_end>",
                "assistant": f"<tool_call>{{\"name\": \"get_xsea_product_list\", \"arguments\": {{}}}}</tool_call><|im_end><|endoftext|>"
            }
        samples.append(sample)

    for _ in range(neg_samples):
        system_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant capable of executing function calls. Available functions: get_xsea_product_list(), get_weather_by_location(location). Respond with a <tool_call> containing a JSON object with 'name' and 'arguments'. For non-function queries, respond with plain text.<|im_end>"
        prompt = random.choice(neg_templates[diversity])
        sample = {
            "prompt": f"{system_prompt}\n<|im_start|>user\n{prompt}<|im_end>",
            "assistant": f"抱歉，我无法识别 '{prompt}' 对应的功能。请澄清或提供其他请求。<|im_end><|endoftext|>"
        }
        samples.append(sample)

    random.shuffle(samples)
    return samples

def save_dataset(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def generate_validation_set():
    val_samples = []
    system_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant capable of executing function calls. Available functions: get_xsea_product_list(), get_weather_by_location(location). Respond with a <tool_call> containing a JSON object with 'name' and 'arguments'. For non-function queries, respond with plain text.<|im_end>"
    
    for _ in range(35):
        loc = random.choice(locations["high"])
        prompt = random.choice(weather_templates["high"]).format(location=loc)
        val_samples.append({
            "prompt": f"{system_prompt}\n<|im_start|>user\n{prompt}<|im_end>",
            "expected": {"name": "get_weather_by_location", "arguments": {"location": loc}}
        })
    for _ in range(35):
        prompt = random.choice(product_templates["high"])
        val_samples.append({
            "prompt": f"{system_prompt}\n<|im_start|>user\n{prompt}<|im_end>",
            "expected": {"name": "get_xsea_product_list", "arguments": {}}
        })
    
    neg_prompts = neg_templates["high"] * 6
    for prompt in random.sample(neg_prompts, 30):
        val_samples.append({
            "prompt": f"{system_prompt}\n<|im_start|>user\n{prompt}<|im_end>",
            "expected": None
        })
    
    random.shuffle(val_samples)
    return val_samples

def post_process_output(output):
    try:
        start = output.find("<tool_call>")
        end = output.find("</tool_call>")
        if start != -1 and end != -1:
            tool_call_str = output[start+11:end]
            tool_call = json.loads(tool_call_str)
            if tool_call["name"] not in valid_functions:
                return f"抱歉，无法识别 '{tool_call['name']}' 对应的功能。<|im_end><|endoftext|>"
            return f"<tool_call>{tool_call_str}</tool_call><|im_end><|endoftext|>"
        return output
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}, Output: {output}")
        return output
    except Exception as e:
        print(f"Post-process error: {e}, Output: {output}")
        return output

def evaluate_function_call(model, tokenizer, validation_set):
    model.eval()
    correct = 0
    total_function_calls = sum(1 for v in validation_set if v["expected"] is not None)
    false_positives = 0
    total_negatives = sum(1 for v in validation_set if v["expected"] is None)

    for i, val in enumerate(validation_set):
        inputs = tokenizer(val["prompt"], return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        decoded_output = post_process_output(decoded_output)
        if i < 10:  # 打印前 10 个样本
            print(f"Sample {i}: Prompt={val['prompt'].split('<|im_start|>user\n')[1].split('<|im_end>')[0]}\nOutput={decoded_output}\nExpected={val['expected']}\n")

        try:
            start = decoded_output.find("<tool_call>")
            end = decoded_output.find("</tool_call>")
            if start != -1 and end != -1:
                tool_call = json.loads(decoded_output[start+11:end])
                if val["expected"] is None:
                    false_positives += 1
                    print(f"False Positive: Got {tool_call}, Expected None")
                elif tool_call == val["expected"]:
                    correct += 1
                else:
                    print(f"Incorrect: Got {tool_call}, Expected {val['expected']}")
            else:
                if val["expected"] is None:
                    correct += 1
                else:
                    print(f"Missed tool call: Got {decoded_output}, Expected {val['expected']}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error in eval: {e}, Output: {decoded_output}")
            if val["expected"] is None:
                correct += 1
            else:
                print(f"Error parsing output: {decoded_output}, Expected {val['expected']}")
        except Exception as e:
            print(f"Eval error: {e}, Output: {decoded_output}")
            if val["expected"] is None:
                correct += 1

    accuracy = correct / len(validation_set)
    recall = correct / total_function_calls if total_function_calls > 0 else 0
    fpr = false_positives / total_negatives if total_negatives > 0 else 0
    print(f"Final: Accuracy={accuracy:.4f}, Recall={recall:.4f}, FPR={fpr:.4f}, Correct={correct}, Total={len(validation_set)}, Total Function Calls={total_function_calls}, False Positives={false_positives}, Total Negatives={total_negatives}")
    return accuracy, recall, fpr

def train_model(dataset_path, output_dir, max_steps):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_model_path,
        max_seq_length=512,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map={"": 0},
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # 增加 LoRA 秩
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=64,  # 增加 alpha
        lora_dropout=0.0,
        use_gradient_checkpointing=True
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="prompt",
        max_seq_length=512,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_steps=20,
            max_steps=max_steps,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=20,
            output_dir=output_dir,
            optim="adamw_torch",
            report_to="none",
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=50,
            gradient_checkpointing=True
        ),
        formatting_func=lambda x: {"text": f"{x['prompt']}\n{x['assistant']}"}  # 明确 prompt 和 assistant 格式
    )
    trainer.train()
    lora_save_path = os.path.join(output_dir, "lora_model")
    model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)
    print(f"Saved LoRA model to {lora_save_path}")
    # 测试预训练和微调模型
    print("Testing pre-trained model:")
    pre_model, pre_tokenizer = FastLanguageModel.from_pretrained(local_model_path, load_in_4bit=True)
    test_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant capable of executing function calls. Available functions: get_xsea_product_list(), get_weather_by_location(location). Respond with a <tool_call> containing a JSON object with 'name' and 'arguments'. For non-function queries, respond with plain text.<|im_end>\n<|im_start|>user\n苏州现在气温多少<|im_end>"
    inputs = pre_tokenizer(test_prompt, return_tensors="pt").to("cuda:0")
    outputs = pre_model.generate(**inputs, max_new_tokens=128, pad_token_id=pre_tokenizer.eos_token_id)
    print(f"Pre-trained output: {pre_tokenizer.decode(outputs[0], skip_special_tokens=False)}")
    del pre_model, pre_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Testing micro-tuned model:")
    model.eval()
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    print(f"Micro-tuned output: {tokenizer.decode(outputs[0], skip_special_tokens=False)}")
    return model, tokenizer

def cleanup_cache():
    cache_dir = "./unsloth_compiled_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleaned up {cache_dir}")
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Cleared PyTorch CUDA cache, allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GiB, reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GiB")

def plot_results(results, x_label, title):
    plt.figure(figsize=(12, 8))
    for metric in ["Accuracy", "Recall", "FPR"]:
        plt.plot(results[x_label], results[metric], marker='o', label=metric)
    plt.xlabel(x_label)
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

valid_functions = ["get_xsea_product_list", "get_weather_by_location"]
def main():
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    validation_set = generate_validation_set()
    print(f"Validation set: {[{'prompt': v['prompt'].split('<|im_start|>user\n')[1].split('<|im_end>')[0], 'expected': v['expected']} for v in validation_set[:5]]}")

    batch_size = 1 * 2
    num_epochs = 2

    size_results = {"Size": [], "Accuracy": [], "Recall": [], "FPR": []}
    for size in [100, 1000, 5000]:
        print(f"Running Dataset Size Experiment: Size={size}")
        dataset = generate_samples(size, diversity="high", neg_ratio=0.3)  # 增加负样本比例
        print(f"Sample prompts (Size={size}): {[s['prompt'].split('<|im_start|>user\n')[1].split('<|im_end>')[0] for s in dataset[:5]]}")
        print(f"Sample assistants: {[s['assistant'] for s in dataset[:5]]}")
        dataset_path = f"dataset_size_{size}.jsonl"
        save_dataset(dataset, dataset_path)
        max_steps = math.ceil(size / batch_size) * num_epochs
        print(f"Calculated max_steps={max_steps} for {num_epochs} epochs")
        output_dir = f"outputs_size_{size}"
        model, tokenizer = train_model(dataset_path, output_dir, max_steps)
        accuracy, recall, fpr = evaluate_function_call(model, tokenizer, validation_set)
        size_results["Size"].append(size)
        size_results["Accuracy"].append(accuracy)
        size_results["Recall"].append(recall)
        size_results["FPR"].append(fpr)
        print(f"Size={size}: Accuracy={accuracy:.4f}, Recall={recall:.4f}, FPR={fpr:.4f}")
        print(f"Size Results: {size_results}")
        del model, tokenizer
        cleanup_cache()

    diversity_results = {"Diversity": [], "Accuracy": [], "Recall": [], "FPR": []}
    for diversity in ["low", "mid", "high"]:
        print(f"Running Sample Diversity Experiment: Diversity={diversity}")
        dataset = generate_samples(1000, diversity=diversity, neg_ratio=0.3)
        print(f"Sample prompts (Diversity={diversity}): {[s['prompt'].split('<|im_start|>user\n')[1].split('<|im_end>')[0] for s in dataset[:5]]}")
        print(f"Sample assistants: {[s['assistant'] for s in dataset[:5]]}")
        dataset_path = f"dataset_diversity_{diversity}.jsonl"
        save_dataset(dataset, dataset_path)
        max_steps = math.ceil(1000 / batch_size) * num_epochs
        print(f"Calculated max_steps={max_steps} for {num_epochs} epochs")
        output_dir = f"outputs_diversity_{diversity}"
        model, tokenizer = train_model(dataset_path, output_dir, max_steps)
        accuracy, recall, fpr = evaluate_function_call(model, tokenizer, validation_set)
        diversity_results["Diversity"].append(diversity)
        diversity_results["Accuracy"].append(accuracy)
        diversity_results["Recall"].append(recall)
        diversity_results["FPR"].append(fpr)
        print(f"Diversity={diversity}: Accuracy={accuracy:.4f}, Recall={recall:.4f}, FPR={fpr:.4f}")
        print(f"Diversity Results: {diversity_results}")
        del model, tokenizer
        cleanup_cache()

    neg_ratio_results = {"Neg Ratio": [], "Accuracy": [], "Recall": [], "FPR": []}
    for neg_ratio in [0.0, 0.1, 0.3]:
        print(f"Running Negative Sample Ratio Experiment: Neg Ratio={neg_ratio}")
        dataset = generate_samples(1000, diversity="high", neg_ratio=neg_ratio)
        print(f"Sample prompts (Neg Ratio={neg_ratio}): {[s['prompt'].split('<|im_start|>user\n')[1].split('<|im_end>')[0] for s in dataset[:5]]}")
        print(f"Sample assistants: {[s['assistant'] for s in dataset[:5]]}")
        dataset_path = f"dataset_neg_ratio_{neg_ratio}.jsonl"
        save_dataset(dataset, dataset_path)
        max_steps = math.ceil(1000 / batch_size) * num_epochs
        print(f"Calculated max_steps={max_steps} for {num_epochs} epochs")
        output_dir = f"outputs_neg_ratio_{neg_ratio}"
        model, tokenizer = train_model(dataset_path, output_dir, max_steps)
        accuracy, recall, fpr = evaluate_function_call(model, tokenizer, validation_set)
        neg_ratio_results["Neg Ratio"].append(neg_ratio)
        neg_ratio_results["Accuracy"].append(accuracy)
        neg_ratio_results["Recall"].append(recall)
        neg_ratio_results["FPR"].append(fpr)
        print(f"Neg Ratio={neg_ratio}: Accuracy={accuracy:.4f}, Recall={recall:.4f}, FPR={fpr:.4f}")
        print(f"Neg Ratio Results: {neg_ratio_results}")
        del model, tokenizer
        cleanup_cache()

    plot_results(size_results, "Size", "Effect of Dataset Size on Metrics")
    plot_results(diversity_results, "Diversity", "Effect of Sample Diversity on Metrics")
    plot_results(neg_ratio_results, "Neg Ratio", "Effect of Negative Sample Ratio on Metrics")

if __name__ == "__main__":
    main()