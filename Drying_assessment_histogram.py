import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
models = ['CanESM5-1', 'EC-Earth3', 'MIROC6', 'MPI-ESM1-2-HR', 'NorESM2-MM']

# Barchart colors (4/5) and (5/5)
color_4of5 = '#fc8d59'  # Lighter orange
color_5of5 = '#e34a33'  # Dark red

all_data = []

# Get data
for scenario in scenarios:
    df = pd.read_excel(rf"C:\Thesis_Merwin\pr_changes\Table_{scenario}_new.xlsx")
    pct_cols = [f'{model} percent change yearly pr' for model in models]

    df['agreement'] = (df[pct_cols] < 0).sum(axis=1)
    df_filtered = df[df['agreement'] >= 4].copy()

    ssp_label = f'{scenario[:3].upper()}-RCP{scenario[3:]}'

    df_melted = df_filtered.melt(id_vars=['agreement'], value_vars=pct_cols,
                                 var_name='Model', value_name='Precip_Change')
    df_melted['Model'] = df_melted['Model'].str.replace(' percent change yearly pr', '')
    df_melted['SSP'] = ssp_label

    all_data.append(df_melted)

master_df = pd.concat(all_data, ignore_index=True)

# X-axis height
global_min = master_df['Precip_Change'].min()
global_max = master_df['Precip_Change'].max()
x_range = global_max - global_min
x_limit_min = global_min - (x_range * 0.05)
x_limit_max = global_max + (x_range * 0.05)

# Layout
fig = plt.figure(figsize=(14, 4))
gs = fig.add_gridspec(1, 4, wspace=0.15)

axs = [fig.add_subplot(gs[0, i]) for i in range(4)]

fig.text(0.02, 0.5, 'Frequency of Projections', va='center', rotation='vertical',
         fontweight='bold', fontsize=12)

# Stacked histogram
bins = np.arange(np.floor(x_limit_min), np.ceil(x_limit_max) + 2, 2)
bin_edges = bins[:-1]

plot_labels = ['a', 'b', 'c', 'd']
global_max_height = 0

for i, scenario in enumerate(scenarios):
    ax = axs[i]
    ssp_label = f'{scenario[:3].upper()}-RCP{scenario[3:]}'
    subset = master_df[master_df['SSP'] == ssp_label]

    data_5 = subset[subset['agreement'] == 5]['Precip_Change']
    data_4 = subset[subset['agreement'] == 4]['Precip_Change']

    counts_5, _ = np.histogram(data_5, bins=bins)
    counts_4, _ = np.histogram(data_4, bins=bins)

    current_max = max(counts_5 + counts_4) if len(counts_5) > 0 else 1
    if current_max > global_max_height:
        global_max_height = current_max

    ax.bar(bin_edges, counts_5, width=2, align='edge',
           color=color_5of5, linewidth=0, zorder=2)

    ax.bar(bin_edges, counts_4, width=2, align='edge', bottom=counts_5,
           color=color_4of5, linewidth=0, zorder=2)

    ax.set_xlim(x_limit_min, x_limit_max)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, zorder=3)
    ax.tick_params(axis='both', labelsize=9)

    formatted_title = f"SSP{scenario[3]}-{scenario[4]}.{scenario[5]}"

    ax.set_title(formatted_title, fontsize=12, fontweight='bold')
    ax.text(0.08, 0.96, plot_labels[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
    ax.set_xlabel('Annual Precipitation Change (%)', fontweight='bold', fontsize=10)

    if i > 0:
        ax.set_yticklabels([])

for ax in axs:
    ax.set_ylim(bottom=0, top=global_max_height * 1.15)

# Legend
legend_elements = [
    Patch(facecolor=color_5of5, label='5/5 Consensus'),
    Patch(facecolor=color_4of5, label='4/5 Consensus')
]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=2, framealpha=0, edgecolor='black')

plt.subplots_adjust(top=0.88, bottom=0.22, left=0.06, right=0.98)
plt.show()