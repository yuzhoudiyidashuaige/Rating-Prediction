import json
import math
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.functional import cosine_similarity

file = "data/All_Beauty.jsonl"
data = []
with open(file, 'r', encoding='utf-8') as fp:
    for line in fp:
        record = json.loads(line.strip())
        data.append(record)

modified_data = []


for record in data:
    modified_data.append({
        'user_id': record['user_id'],  # ????
        'book_id': record['asin'],  # ????
        'rating': record['rating']  # ??????? 'overall'
    })

df = pd.DataFrame(modified_data)

from collections import defaultdict
from sklearn.model_selection import train_test_split

usersPerItem = defaultdict(set)  # Maps an item to the users who rated it
itemsPerUser = defaultdict(set)  # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {}  # To retrieve a rating for a specific user/item pair

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

for _, d in train_set.iterrows():
    book_id = d['book_id']
    user_id = d['user_id']
    rating = d['rating']

    usersPerItem[book_id].add(user_id)
    itemsPerUser[user_id].add(book_id)
    reviewsPerUser[user_id].append(d.to_dict())
    reviewsPerItem[book_id].append(d.to_dict())
    ratingDict[(user_id, book_id)] = rating

item_mean_rating = {}
mean_rating = train_set['rating'].mean()

print("????:", mean_rating)

# for key in reviewsPerItem:
#     ratings = [review['rating'] for review in reviewsPerItem[key]]  # ???? review ? rating
#     print(ratings)
#     item_mean_rating[key] = sum(ratings) / len(ratings)

for item in usersPerItem:
    rating_list = [ratingDict[(user, item)] for user in usersPerItem[item]]
    item_mean_rating[item] = sum(rating_list) / len(rating_list)


def Cosine(i1, i2):
    # Between two items
    inter = usersPerItem[i1].intersection(usersPerItem[i2])
    numer = 0
    denom1 = 0
    denom2 = 0
    for u in inter:
        numer += ratingDict[(u,i1)]*ratingDict[(u,i2)]
    for u in usersPerItem[i1]:
        denom1 += ratingDict[(u,i1)]**2
    for u in usersPerItem[i2]:
        denom2 += ratingDict[(u,i2)]**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom

def predict_rating(user, item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        item1 = d['book_id']
        if item1 == item: continue
        ratings.append(d['rating'] - item_mean_rating[item1])
        similarities.append(Cosine(item, item1))
    if sum(similarities) > 0:
        weightedRatings = [(x * y) for x, y in zip(ratings, similarities)]
        return item_mean_rating[item] + sum(weightedRatings) / sum(similarities)
    else:
        return mean_rating


user_mean_rating = {}
for user in itemsPerUser:
    rating_list = [ratingDict[(user, item)] for item in itemsPerUser[user]]
    user_mean_rating[user] = sum(rating_list) / len(rating_list)



def MSE(predictions, labels):
    differences = [(x - y) ** 2 for x, y in zip(predictions, labels)]
    return sum(differences) / len(differences)


predict_results = [
    predict_rating(row['user_id'], row['book_id']) for _, row in test_set.iterrows()
]

# predict_results=numpy.array(predict_results)
actual = [int(d['rating']) for _, d in test_set.iterrows()]
# actual=numpy.array(actual)
mse = MSE(predict_results, actual)
print("mse:", mse)