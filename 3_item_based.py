import pandas as pd
import numpy as np
import os
import time
from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
from precision_recall import precision_recall_at_k_2

df = pd.read_csv('data/processed/ratings_clean.csv')
print('-'*20)
print(f'\nLoaded {len(df):,} ratings')
print()

print('-'*20)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

print('Starting Item-based model:')
print()
print('Using GridSearchCV for the best parameters')
print()

start_time = time.time()
param_grid = {
    'k': [10, 20, 40],
    'min_k': [1, 3, 5],
    'sim_options': {
        'name': ['cosine', 'pearson'],
        'user_based': [False]
    }
}
gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
gs.fit(data)
print('Finished searching')
print()

print('-'*20)
best = gs.best_params['rmse']
print()
print(f'The best parameters configuration is:')
print(f'  - k (neighbors): {best["k"]}')
print(f'  - min_k: {best["min_k"]}')
print(f'  - similarity: {best["sim_options"]["name"]}')
print()

print('-'*20)
print("\nTraining final model with best parameters")
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
print('Train test split: 80/20')
print(f'Training set: {trainset.n_ratings:,} ratings')
print(f'Test set: {len(testset):,} ratings')
print()

sim_options = {
    'name': best['sim_options']['name'],
    'user_based': False
}

print('-'*10)
print('Running model:')
model = KNNWithMeans(
    k = best['k'],
    min_k = best['min_k'],
    sim_options = sim_options
)

model.fit(trainset)
predictions = model.test(testset)

end_time = time.time()
elapsed = round(end_time - start_time, 4)

print('Finished')
print()

print('-'*20)
print('Evaluating model:')
rmse = accuracy.rmse(predictions, verbose=False)
mae = accuracy.mae(predictions,  verbose=False)

p5,  r5  = precision_recall_at_k_2(predictions, k=5,  threshold=3.5)
p10, r10 = precision_recall_at_k_2(predictions, k=10, threshold=3.5)

print('Item-based performance score:')
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
    'Model': 'Item-based (KNN)',
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
print('Item-Based model completed')