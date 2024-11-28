import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

file = "All_Beauty.jsonl"
data = []
with open(file, 'r', encoding='utf-8') as fp:
    for line in fp:
        record = json.loads(line.strip())
        data.append(record)

features = []
ratings = []
for record in data:
    text_exclamation_count = record["text"].count('!')
    title_exclamation_count = record["title"].count('!')
    features.append([text_exclamation_count, title_exclamation_count])
    ratings.append(record["rating"])

X = np.array(features)
y = np.array(ratings)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"Mean Squared Error: {mse}")
