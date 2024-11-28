import json

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Step 1: 加载数据
file = "All_Beauty.jsonl"
data = []

with open(file, 'r', encoding='utf-8') as fp:
    for line in fp:
        record = json.loads(line.strip())
        # 检查必须字段是否存在
        if "user_id" in record and "asin" in record and "rating" in record:
            data.append((record["user_id"], record["asin"], record["rating"]))

print(f"Total records loaded: {len(data)}")

# Step 2: 构建 Surprise 的 Dataset 对象
reader = Reader(rating_scale=(1, 5))  # 假设评分范围是 1 到 5
dataset = Dataset.load_from_df(pd.DataFrame(data, columns=["user_id", "asin", "rating"]), reader)

# Step 3: 划分训练集和测试集
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# Step 4: 初始化 SVD 模型并训练
model = SVD()
model.fit(trainset)

# Step 5: 测试模型并评估
predictions = model.test(testset)
# print(f"RMSE: {rmse(predictions)}")

sse = 0
for p in predictions:
    sse += (p.r_ui - p.est)**2

print(sse / len(predictions))
