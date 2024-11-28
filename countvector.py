import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

file = "All_Beauty.jsonl"
data = []

with open(file, 'r', encoding='utf-8') as fp:
    for line in fp:
        record = json.loads(line.strip())
        data.append(record)

df = pd.DataFrame(data)
df = df[['text', 'rating']].dropna()

vectorizer = CountVectorizer(max_features=1000)  # 限制最大特征数量，避免内存问题
X = vectorizer.fit_transform(df['text']).toarray()
y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(solver = 'newton-cg')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")
