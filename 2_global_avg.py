import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from precision_recall import precision_recall_at_k_1

df = pd.read_csv('data/processed/ratings_clean.csv')
print('-'*20)
print(f'Loaded {len(df):,} ratings')
print()

print('-'*20)
print('Starting Global average model:')
print()
train, test = train_test_split(df, test_size=0.2, random_state=42)
print('Train test split: 80/20')
print(f'Training set: {len(train):,} ratings')
print(f'Test set: {len(test):,} ratings')
print()

print('-'*20)
print('Running model:')
start_time = time.time()

global_mean = train['rating'].mean()
predictions = np.full(len(test), global_mean)
actual = test['rating'].values

end_time = time.time()
elapsed = round(end_time - start_time, 4)

print('Finished')
print(f'The global avg rating is: {global_mean:.4f}')
print()

print('-'*20)
print('Evaluating model:')
rmse = np.sqrt(mean_squared_error(actual, predictions))
mae = mean_absolute_error(actual, predictions)

p5, r5 = precision_recall_at_k_1(test, global_mean, k=5,  threshold=3.5)
p10, r10 = precision_recall_at_k_1(test, global_mean, k=10, threshold=3.5)

print('Global average performance score:')
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

results = pd.DataFrame([{
    'Model': 'Global Average (Baseline)',
    'RMSE': round(rmse,4),
    'MAE': round(mae,4),
    'Precision@5': round(p5,4),
    'Recall@5': round(r5,4),
    'Precision@10': round(p10,4),
    'Recall@10': round(r10,4),
    'Time (s)': elapsed
}])

results.to_csv('results/performance_metrics.csv', index=False)
print('Results saved to results/performance_metrics.csv')
print('Global average baseline model completed')

