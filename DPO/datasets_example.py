from datasets import load_dataset

dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
print(dataset.column_names)  # 打印字段名
print(dataset[0])  # 打印第一条样本

# 打印数据集的大小
print("数据集的大小：", len(dataset))
