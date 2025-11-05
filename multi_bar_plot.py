import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import textwrap
from csv_loader import load_and_average_data

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

max_line_length = 20
categories, methods, fw_times, bw_times = load_and_average_data()

for i, method in enumerate(methods):
    all_comps = method.split("_")
    res = []
    for comp in all_comps:
        if not comp: continue
        res.append(comp.capitalize())
    methods[i] = " ".join(res)

wrapper_categories = []
for category in categories:
    wrapped = textwrap.fill(category, width=max_line_length)
    wrapper_categories.append(wrapped)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
fig.suptitle("Forward and Backward Time Performance Comparison", fontweight='bold')

colors = ['#4C72B0', '#55A868', '#555555', '#C44E52']

y_pos = np.arange(len(categories))
bar_width = 0.15
bar_spacing = 0.02

ax1.set_title("Forward Time (ms)", pad=10)
ax1.set_xlabel("Time (ms)")
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.set_title("Backward Time (ms)", pad=10)
ax2.set_xlabel("Time (ms)")
ax2.grid(True, linestyle='--', alpha=0.7)

max_fw = max(max(vals) for vals in fw_times.values())
max_bw = max(max(vals) for vals in bw_times.values())
ax1.set_xlim(0, max_fw * 1.25)
ax2.set_xlim(0, max_bw * 1.25)

for i, category in enumerate(categories):
    fw_values = fw_times[category]
    min_fw = min(fw_values)
    
    bw_values = bw_times[category]
    min_bw = min(bw_values)
    
    # percentage w.r.t baseline
    fw_percent = [0] + [((fw_values[j] - fw_values[0]) / fw_values[0]) * 100 for j in range(1, len(fw_values))]
    bw_percent = [0] + [((bw_values[j] - bw_values[0]) / bw_values[0]) * 100 for j in range(1, len(bw_values))]
    
    for j, method in enumerate(methods):
        bar_y = y_pos[i] + j * (bar_width + bar_spacing)
        
        bar_fw = ax1.barh(
            bar_y, fw_values[j], 
            height=bar_width, 
            color=colors[j],
            edgecolor='black',
            linewidth=0.7
        )
        
        bar_bw = ax2.barh(
            bar_y, bw_values[j], 
            height=bar_width, 
            color=colors[j],
            edgecolor='black',
            linewidth=0.7
        )
        
        label_x = fw_values[j] + max_fw * 0.01
        fontweight = 'bold' if fw_values[j] == min_fw else 'normal'
        ax1.text(
            label_x, bar_y, 
            f"{fw_values[j]:.2f}", 
            va='center', ha='left',
            fontweight=fontweight
        )
        
        label_x = bw_values[j] + max_bw * 0.01
        fontweight = 'bold' if bw_values[j] == min_bw else 'normal'
        ax2.text(
            label_x, bar_y, 
            f"{bw_values[j]:.2f}", 
            va='center', ha='left',
            fontweight=fontweight
        )
        
        if j > 0:
            percent_text = f"{fw_percent[j]:+.1f}%"
            color = '#ffffff' if fw_percent[j] > 0 else '#000000'
            ax1.text(
                fw_values[j] * 0.5, bar_y, 
                percent_text, 
                va='center', ha='center',
                color=color, fontweight='bold',
                fontsize=9
            )
            
            percent_text = f"{bw_percent[j]:+.1f}%"
            color = '#ffffff' if bw_percent[j] > 0 else '#000000'
            ax2.text(
                bw_values[j] * 0.5, bar_y, 
                percent_text, 
                va='center', ha='center',
                color=color, fontweight='bold',
                fontsize=9
            )

ax1.set_yticks(y_pos + bar_width + bar_spacing)
ax1.set_yticklabels(wrapper_categories)
ax1.invert_yaxis()

legend_handles = [
    plt.Rectangle((0,0),1,1, color=colors[i], ec='black') 
    for i in range(len(methods))
]

fig.legend(
    legend_handles, methods, 
    title="Methods", 
    loc='upper center',
    ncol=len(methods),
    bbox_to_anchor=(0.5, 0.96),
    frameon=True,
    framealpha=0.9
)

plt.subplots_adjust(top=0.85, bottom=0.12)

fig.text(
    0.5, 0.04, 
    "Note: Bold values indicate best performance in each category. "
    "Percentage changes are relative to Baseline method.",
    ha='center', fontsize=9, alpha=0.8
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('Test.png', dpi=320)