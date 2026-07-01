import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
models = ['CanESM5-1', 'EC-Earth3', 'MIROC6', 'MPI-ESM1-2-HR', 'NorESM2-MM']
baseline = '2015-2024'
future = '2045-2054'


# Colors
model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# print data per ssp
scenario = 'ssp585'
df_print = pd.read_excel(rf"C:\Thesis_Merwin\pr_changes\Table_{scenario}_new.xlsx")

for model in models:
    data_baseline = df_print[f'{model} avg yearly pr in mm for {baseline}']
    data_future = df_print[f'{model} avg yearly pr in mm for {future}']
    print(f"{model} baseline mean =", data_baseline.mean())
    print(f"{model} future mean =", data_future.mean())
    print(f"{model} increase =", (data_future.mean() - data_baseline.mean()) / data_baseline.mean() * 100, "%")

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

plot_labels = ['a', 'b', 'c', 'd']

for i, scenario in enumerate(scenarios):
    ax = axs[i]

    df = pd.read_excel(rf"C:\Thesis_Merwin\pr_changes\Table_{scenario}_new.xlsx")

    baseline_data = []
    future_data = []

    for model in models:
        base_col = df[f'{model} avg yearly pr in mm for {baseline}'].dropna().values
        fut_col = df[f'{model} avg yearly pr in mm for {future}'].dropna().values

        baseline_data.append(base_col)
        future_data.append(fut_col)

    positions = np.arange(1, len(models) + 1) * 3
    base_pos = positions - 0.5
    fut_pos = positions + 0.5

    # Baseline Boxplots
    bplot1 = ax.boxplot(baseline_data, positions=base_pos, widths=0.8,
                        patch_artist=True,
                        showmeans=True,
                        meanline=True,
                        meanprops=dict(linestyle='--', linewidth=1.5, color='black'),
                        boxprops=dict(alpha=0.4, color='black'),
                        medianprops=dict(color='black', linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))

    # Future Boxplots
    bplot2 = ax.boxplot(future_data, positions=fut_pos, widths=0.8,
                        patch_artist=True,
                        showmeans=True,
                        meanline=True,
                        meanprops=dict(linestyle='--', linewidth=1.5, color='black'),
                        boxprops=dict(alpha=1.0, color='black'),
                        medianprops=dict(color='black', linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))

    # Set colors
    for patch, color in zip(bplot1['boxes'], model_colors):
        patch.set_facecolor(color)
    for patch, color in zip(bplot2['boxes'], model_colors):
        patch.set_facecolor(color)

    # Layout formatting
    formatted_title = f"SSP{scenario[3]}-{scenario[4]}.{scenario[5]}"
    ax.set_title(formatted_title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_xlim(1, positions[-1] + 2)

    if i == 0 or i == 2:
        ax.set_ylabel('Annual Precipitation (mm)', fontweight='bold', fontsize=11)

    ax.text(0.95, 0.95, plot_labels[i], transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='right')

fig.tight_layout()

fig.subplots_adjust(bottom=0.20, hspace=0.13)

# Custom legend
leg_ax = fig.add_axes([0.1, 0.09, 0.89, 0.09])
leg_ax.axis('off')
leg_ax.set_xlim(0, 1)
leg_ax.set_ylim(0, 1)

x_cols = np.linspace(0, 0.92, 7)
y_row1 = 0.83  # Height top row
y_row2 = 0.56  # Height middle row
y_row3 = 0.27  # Height bottom row

box_w = 0.05  # Color box width
box_h = 0.19  # Color box height

label_x = x_cols[1] - 0.05

leg_ax.text(label_x, y_row2, baseline, va='center', ha='right', fontsize=10, fontweight='bold')
leg_ax.text(label_x, y_row3, future, va='center', ha='right', fontsize=10, fontweight='bold')

# Add legend items
for idx, (model, color) in enumerate(zip(models, model_colors)):
    cx = x_cols[idx + 1]
    leg_ax.text(cx, y_row1, model, va='center', ha='center', fontsize=10, fontweight='bold')
    leg_ax.add_patch(Rectangle((cx - box_w / 2, y_row2 - box_h / 2), box_w, box_h,
                               facecolor=color, alpha=0.4, edgecolor='black'))
    leg_ax.add_patch(Rectangle((cx - box_w / 2, y_row3 - box_h / 2), box_w, box_h,
                               facecolor=color, alpha=1.0, edgecolor='black'))

# Mean and median lines
cx = x_cols[6]
leg_ax.plot([cx - 0.08, cx - 0.04], [y_row2, y_row2], color='black', linestyle='--', linewidth=2)
leg_ax.text(cx - 0.03, y_row2, 'Mean', va='center', ha='left', fontsize=10)
leg_ax.plot([cx - 0.08, cx - 0.04], [y_row3, y_row3], color='black', linestyle='-', linewidth=2)
leg_ax.text(cx - 0.03, y_row3, 'Median', va='center', ha='left', fontsize=10)

plt.show()
