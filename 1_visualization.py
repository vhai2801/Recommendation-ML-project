import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

os.makedirs('results/figures', exist_ok=True)

col_names = ['userId', 'movieId', 'categoryId', 'reviewId', 'rating', 'reviewDate']
df = pd.read_csv('data/raw/movie-ratings.txt', names=col_names)

n_users = df['userId'].nunique()
n_movies = df['movieId'].nunique()
n_ratings = len(df)
sparsity = 1 - (n_ratings / (n_users * n_movies))

print('_'*20)
print(f"Number of Users: {n_users:,}")
print(f"Number of Movies: {n_movies:,}")
print(f"Number of Ratings: {n_ratings:,}")
print(f"Matrix Sparsity: {sparsity:.4%}")
print()

print('_'*20)
print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")
print(f"Unique ratings: {sorted(df['rating'].unique().tolist())}")

before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"\nRows before dedup: {before:,}")
print(f"Rows after dedup: {after:,}")
print(f"Removed:{before - after:,}")
print()

print('_'*20)
print('Visualizing chart number 1:')

rating_counts = df['rating'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(rating_counts.index, rating_counts.values, color='#4472C4', edgecolor='white', width=0.6)
for bar, val in zip(bars, rating_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:,}', ha='center', va='bottom', fontsize=10)
ax.set_xlabel('Rating', fontsize=12)
ax.set_ylabel('Number of Ratings', fontsize=12)
ax.set_title('Distribution of Movie Ratings', fontsize=14, fontweight='bold')
ax.set_xticks([1, 2, 3, 4, 5])
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('results/figures/distribution-chart.png', dpi=150, bbox_inches='tight', facecolor='white')

print('Exported chart 1 as "distribution-chart.png" and saved to results/figures')
print()

print('_'*20)
print('Visualizing chart number 2:')

ratings_per_user  = df.groupby('userId')['rating'].count().sort_values(ascending=False)
ratings_per_movie = df.groupby('movieId')['rating'].count().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, data, color, xlabel, title in [
    (axes[0], ratings_per_user,  '#4472C4', 'Users (sorted by activity)',    f'Ratings per User (median={ratings_per_user.median():.0f}, max={ratings_per_user.max()})'),
    (axes[1], ratings_per_movie, '#E06C5A', 'Movies (sorted by popularity)', f'Ratings per Movie (median={ratings_per_movie.median():.0f}, max={ratings_per_movie.max()})'),
]:
    x = range(len(data))
    ax.fill_between(x, data.values, color=color, alpha=0.75)
    ax.plot(x, data.values, color=color, linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Number of Ratings (log scale)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('results/figures/tail-chart.png', dpi=150, bbox_inches='tight', facecolor='white')

print('Exported chart 2 as "tail-chart.png" and saved to results/figures')
print()

print('_'*20)
print('Visualizing chart number 3:')

cat_counts = df['categoryId'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, max(4, len(cat_counts) * 0.35)))
bars = ax.barh(cat_counts.index.astype(str), cat_counts.values, color='#5BA85F', edgecolor='white')
for bar, val in zip(bars, cat_counts.values):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9)
ax.set_xlabel('Number of Ratings', fontsize=11)
ax.set_ylabel('Category ID', fontsize=11)
ax.set_title('Ratings by Category', fontsize=14, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('results/figures/category-chart.png', dpi=150, bbox_inches='tight', facecolor='white')

print('Exported chart 3 as "category-chart.png" and saved to results/figures')
print()

print('-'*20)
df_clean = df[['userId', 'movieId', 'rating']].copy()
df_clean.to_csv('data/processed/ratings_clean.csv', index=False)

print(f"Cleaned data saved to: data/processed/ratings_clean.csv")
print(f"Shape: {df_clean.shape}")