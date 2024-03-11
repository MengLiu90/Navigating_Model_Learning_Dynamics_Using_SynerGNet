import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# bac trend plot
figure_dir = ''
df = pd.read_csv('for_plots.csv')

fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
ax.plot(df.power, df.mean_bac, color='blue', lw=2, marker='o', label='Mean Values')
# Create error bar rectangles
for x, y, err in zip(df['power'], df['mean_bac'], df['bac_std']):
    error_bar = patches.Rectangle((x - 0.08, y - err), 0.16, 2 * err, linewidth=1, edgecolor='blue', facecolor='none')
    ax.add_patch(error_bar)

# plt.legend(['Test loss'], loc='lower right')
ax.set_xlabel(''r'$2^n$', fontsize=18)
ax.set_xticks(df.power)
ax.set_xticklabels(['orginal data' if x == -1 else x for x in df.power])
ax.set_ylabel('BAC', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(figure_dir, 'bac_trend_box_error.png'))
plt.show()

# loss trend plot
fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
ax.plot(df.power, df.mean_loss, color='m', lw=2, marker='o')
# Create error bar rectangles
for x, y, err in zip(df['power'], df['mean_loss'], df['loss_std']):
    error_bar = patches.Rectangle((x - 0.08, y - err), 0.16, 2 * err, linewidth=1, edgecolor='m', facecolor='none')
    ax.add_patch(error_bar)
ax.set_xlabel(''r'$2^n$', fontsize=18)
ax.set_xticks(df.power)
ax.set_xticklabels(['orginal data' if x == -1 else x for x in df.power])
plt.ylabel('Loss', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(figure_dir, 'loss_trend_box_error.png'))
plt.show()

