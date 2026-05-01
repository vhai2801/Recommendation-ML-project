import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

os.makedirs('results/figures', exist_ok=True)

metrics = pd.read_csv('results/performance_metrics.csv')
short_labels = ['Baseline', 'Item-CF', 'Trust-CF']
colors = ['#888780', '#4472C4', '#E06C5A']
x = range(len(metrics))
w = 0.4

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')

for ax, metric, title, ylim in [
    (axes[0], 'RMSE', 'RMSE (lower is better)', (0.9,  metrics['RMSE'].max() * 1.15)),
    (axes[1], 'MAE',  'MAE (lower is better)',  (0.6,  metrics['MAE'].max()  * 1.15)),
]:
    bars = ax.bar(x, metrics[metric], width=w, color=colors, edgecolor='white')
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    for bar, val in zip(bars, metrics[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

legend_handles = [mpatches.Patch(color=c, label=m) for c, m in zip(colors, short_labels)]
fig.legend(handles=legend_handles, loc='upper center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, 1.02), frameon=False)
plt.tight_layout()
plt.savefig('results/figures/rmse-mae-comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')

bw = 0.25
for ax, m5, m10, title in [
    (axes[0], 'Precision@5', 'Precision@10', 'Precision@K (higher is better)'),
    (axes[1], 'Recall@5',    'Recall@10',    'Recall@K (higher is better)'),
]:
    all_vals = list(metrics[m5]) + list(metrics[m10])
    y_min = min(all_vals) * 0.5
    y_max = 1.3

    x5  = [i - bw/2 for i in x]
    x10 = [i + bw/2 for i in x]
    bars5  = ax.bar(x5,  metrics[m5],  width=bw, color=colors, edgecolor='white', label='K=5')
    bars10 = ax.bar(x10, metrics[m10], width=bw, color=colors, edgecolor='white', alpha=0.55, label='K=10')

    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Score (log scale)', fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=10, frameon=False)

    for bar, val in zip(list(bars5) + list(bars10), list(metrics[m5]) + list(metrics[m10])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

legend_handles = [mpatches.Patch(color=c, label=m) for c, m in zip(colors, short_labels)]
fig.legend(handles=legend_handles, loc='upper center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, 1.02), frameon=False)
plt.tight_layout()
plt.savefig('results/figures/precision-recall-comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

metrics['F1@5']  = 2 * metrics['Precision@5']  * metrics['Recall@5']  / (metrics['Precision@5']  + metrics['Recall@5'])
metrics['F1@10'] = 2 * metrics['Precision@10'] * metrics['Recall@10'] / (metrics['Precision@10'] + metrics['Recall@10'])

metrics['Efficiency@5']  = metrics['F1@5']  / metrics['Time (s)']
metrics['Efficiency@10'] = metrics['F1@10'] / metrics['Time (s)']

m2 = metrics[metrics['Model'] != 'Global Average (Baseline)'].reset_index(drop=True)
labels2 = ['Item-CF', 'Trust-CF']
colors2 = ['#4472C4', '#E06C5A']
x2 = range(len(m2))

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('white')

bw = 0.25
x5  = [i - bw/2 for i in x2]
x10 = [i + bw/2 for i in x2]
bars5  = ax.bar(x5,  m2['Efficiency@5'],  width=bw, color=colors2, edgecolor='white', label='K=5')
bars10 = ax.bar(x10, m2['Efficiency@10'], width=bw, color=colors2, edgecolor='white', alpha=0.55, label='K=10')

ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.3f}' if v < 1 else f'{v:.1f}'))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

all_vals = list(m2['Efficiency@5']) + list(m2['Efficiency@10'])
ax.set_ylim(min(all_vals) * 0.3, max(all_vals) * 5)

ax.set_title('Efficiency: F1 / Training Time (higher is better)', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('F1 per second (log scale)', fontsize=11)
ax.set_xticks(list(x2))
ax.set_xticklabels(labels2, fontsize=10)
ax.spines[['top', 'right']].set_visible(False)

for bar, val in zip(list(bars5) + list(bars10), list(m2['Efficiency@5']) + list(m2['Efficiency@10'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
            f'{val:.4f}' if val < 1 else f'{val:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

legend_handles = [mpatches.Patch(color=c, label=m) for c, m in zip(colors2, labels2)]
ax.legend(handles=legend_handles, fontsize=10, frameon=False)
plt.tight_layout(pad=2.0)
plt.savefig('results/figures/efficiency-comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


print('Finished creating 3 charts. Results are saved to results/figures')