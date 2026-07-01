import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

scenarios = ['ssp585', 'ssp370', 'ssp245', 'ssp126']
models = ['CanESM5-1', 'EC-Earth3', 'MIROC6', 'MPI-ESM1-2-HR', 'NorESM2-MM']
categories = [0, 1, 2, 3, 4, 5]

counts_per_category = {cat: [] for cat in categories}

# Extract data per ssp
for scenario in scenarios:
    df = pd.read_excel(rf"C:\Thesis_Merwin\pr_changes\Table_{scenario}_new.xlsx")

    pct_cols = [f'{model} percent change yearly pr' for model in models]

    # Calculate agreement and count basins
    agreement_scores = (df[pct_cols] < 0).sum(axis=1)
    basin_counts = agreement_scores.value_counts().reindex(categories, fill_value=0)

    for cat in categories:
        counts_per_category[cat].append(basin_counts[cat])

# Plotting
fig, ax = plt.subplots(figsize=(12, 5))

y_labels = [f"SSP{scenario[3]}-{scenario[4]}.{scenario[5]}" for scenario in scenarios]

# Colors
stack_colors = ['#4292c6', '#9ecae1', '#c6dbef', '#fcbb84', '#fc8d59', '#e34a33']

left = np.zeros(len(scenarios))

for i, cat in enumerate(categories):
    values = counts_per_category[cat]

    bars = ax.barh(y_labels, values, left=left, label=f'{cat}/5 Drying',
                  color=stack_colors[i], edgecolor='black', height=0.6)

    ax.bar_label(bars, label_type='center', fontsize=11, color='black', fontweight='medium',
                 fmt=lambda x: f'{int(x)}' if x > 0 else '')

    left += np.array(values)

# Positions of the ticks
ticks = [50, 100, 150, 200, 250, 292]
labels = ['50', '100', '150', '200', '250', '292']

ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.set_yticklabels(y_labels, fontweight='bold')
ax.set_xlim(0, 292)
ax.set_xlabel('Number of Drainage Basins', fontsize=12, fontweight='bold', labelpad=5)

ax.legend(title='',
          bbox_to_anchor=(0.5, -0.14),
          loc='upper center',
          ncol=6,
          frameon=False)

fig.tight_layout()

plt.show()