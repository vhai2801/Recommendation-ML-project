import pandas as pd
import numpy as np
import os
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from precision_recall import precision_recall_at_k_3

df = pd.read_csv('data/processed/ratings_clean.csv')
print('-'*20)
print(f'\nLoaded {len(df):,} ratings')
print()

trust_cols = ['trustorId', 'trusteeId', 'trustRating']
trusts = pd.read_csv('data/raw/trusts.txt', names=trust_cols)
print(f'Loaded {len(trusts):,} trust links/')
print()

trust_dict = defaultdict(dict)
for _, row in trusts.iterrows():
    trust_dict[row['trustorId']][row['trusteeId']] = row['trustRating']

print(f'Trust network built for {len(trust_dict):,} users')
print()

print('-'*20)
print('Starting Trust network model:')
print()
train, test = train_test_split(df, test_size=0.2, random_state=42)
print('Train test split: 80/20')
print(f'Training set: {len(train):,} ratings')
print(f'Test set: {len(test):,} ratings')
print()

user_ratings = defaultdict(dict)
for _, row in train.iterrows():
    user_ratings[row['userId']][row['movieId']] = row['rating']

global_mean = train['rating'].mean()
print()

def predict_trust_based(userId, movieId, trust_dict, user_ratings, global_mean):
    trusted_users = trust_dict.get(userId, {})
    
    weighted_sum = 0
    weight_total = 0
    
    for trusted_uid, trust_score in trusted_users.items():
        if movieId in user_ratings.get(trusted_uid, {}):
            rating = user_ratings[trusted_uid][movieId]
            weighted_sum += trust_score * rating
            weight_total += trust_score
    
    if weight_total > 0:
        return weighted_sum / weight_total
    else:
        return global_mean
    

start_time = time.time()

actuals     = []
predictions = []

for _, row in test.iterrows():
    pred = predict_trust_based(
        row['userId'], row['movieId'],
        trust_dict, user_ratings, global_mean
    )
    predictions.append(pred)
    actuals.append(row['rating'])

predictions = np.array(predictions)
actuals = np.array(actuals)

end_time = time.time()
elapsed  = round(end_time - start_time, 4)

print('Finished')
print(f'The global avg rating is: {global_mean:.4f}')
print()

print('-'*20)
print('Evaluating model:')
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae  = mean_absolute_error(actuals, predictions)

p5,  r5  = precision_recall_at_k_3(test, predictions, k=5,  threshold=3.5)
p10, r10 = precision_recall_at_k_3(test, predictions, k=10, threshold=3.5)

print('Trust network performance score:')
print()
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print()
print(f'Precision@5: {p5:.4f}')
print(f'Recall@5: {r5:.4f}')
print()
print(f'Precision@10: {p10:.4f}')
print(f'Recall@10: {r10:.4f}')
print()
print(f'Time elapsed: {elapsed} sec')
print()

print('-'*20)
print('Exporting results:')
os.makedirs('results', exist_ok=True)
existing = pd.read_csv('results/performance_metrics.csv')

new_row = pd.DataFrame([{
    'Model': 'Trust Network (user-based)',
    'RMSE': round(rmse,4),
    'MAE': round(mae,4),
    'Precision@5': round(p5,4),
    'Recall@5': round(r5,4),
    'Precision@10': round(p10,4),
    'Recall@10': round(r10,4),
    'Time (s)': elapsed
}])

results = pd.concat([existing, new_row], ignore_index=True)
results.to_csv('results/performance_metrics.csv', index=False)
print('Results saved to results/performance_metrics.csv')
print('Trust network user-based model completed')