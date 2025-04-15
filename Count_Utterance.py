# from textgrid import TextGrid
# import os

# def count_utterances_in_textgrid(tg_path):
#     tg = TextGrid.fromFile(tg_path)
#     count = 0
#     for tier in tg.tiers:
#         if not hasattr(tier, 'intervals'):
#             continue  # skip point tiers
#         count += len([
#             i for i in tier.intervals 
#             if i.mark.strip() not in ["", "SIL", "⟨laughter⟩"]
#         ])
#     return count

# def count_all_textgrids(folder_path):
#     total_utterances = 0
#     file_counts = {}

#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith(".TextGrid"):
#                 full_path = os.path.join(root, file)
#                 count = count_utterances_in_textgrid(full_path)
#                 file_counts[file] = count
#                 total_utterances += count

#     return total_utterances, file_counts

# # === USAGE ===
# folder_path = "."  # <-- update this path
# total, breakdown = count_all_textgrids(folder_path)

# print(f"\nTotal utterances across all files: {total}")
# print("\nPer file breakdown:")
# # for fname, count in sorted(breakdown.items()):
# #     print(f"{fname}: {count}")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 10))

# Set background color
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

# Define colors
colors = {
    'input': ('#d1e7f0', '#2c88b9'),
    'intermediate': ('#d8e8d5', '#548235'),
    'stance': ('#d5c8e7', '#674ea7'),
    'random_forest': ('#ffe6cc', '#d79b00'),
    'output': ('#f8cecc', '#b85450')
}

# Remove axes
ax.axis('off')

# Add title
ax.text(0.5, 0.95, 'Multimodal Dialogue Act Classification with Random Forest', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# Create boxes
# Input Features
input_box = patches.Rectangle((0.05, 0.65), 0.15, 0.25, 
                             facecolor=colors['input'][0], 
                             edgecolor=colors['input'][1],
                             linewidth=2, alpha=0.9, zorder=1,
                             label='Input Features')

# Audio Features
audio_box = patches.Rectangle((0.07, 0.8), 0.11, 0.08, 
                             facecolor='#a8d5e5', 
                             edgecolor=colors['input'][1],
                             linewidth=1, alpha=0.9, zorder=2)

# Textual Features
text_box = patches.Rectangle((0.07, 0.67), 0.11, 0.08, 
                             facecolor='#a8d5e5', 
                             edgecolor=colors['input'][1],
                             linewidth=1, alpha=0.9, zorder=2)

# Intermediate Representations
inter_box = patches.Rectangle((0.3, 0.15), 0.2, 0.75, 
                             facecolor=colors['intermediate'][0], 
                             edgecolor=colors['intermediate'][1],
                             linewidth=2, alpha=0.9, zorder=1,
                             label='Intermediate Representations')

# VAD Estimation
vad_box = patches.Rectangle((0.32, 0.73), 0.16, 0.15, 
                           facecolor='#c5e0b4', 
                           edgecolor=colors['intermediate'][1],
                           linewidth=1, alpha=0.9, zorder=2)

# Categorical Emotion
emotion_box = patches.Rectangle((0.32, 0.55), 0.16, 0.15, 
                               facecolor='#c5e0b4', 
                               edgecolor=colors['intermediate'][1],
                               linewidth=1, alpha=0.9, zorder=2)

# Certainty Estimation
certainty_box = patches.Rectangle((0.32, 0.37), 0.16, 0.15, 
                                 facecolor='#c5e0b4', 
                                 edgecolor=colors['intermediate'][1],
                                 linewidth=1, alpha=0.9, zorder=2)

# Engagement Estimation
engagement_box = patches.Rectangle((0.32, 0.17), 0.16, 0.15, 
                                  facecolor='#c5e0b4', 
                                  edgecolor=colors['intermediate'][1],
                                  linewidth=1, alpha=0.9, zorder=2)

# Stance Vector
stance_box = patches.Rectangle((0.58, 0.63), 0.16, 0.25, 
                              facecolor=colors['stance'][0], 
                              edgecolor=colors['stance'][1],
                              linewidth=2, alpha=0.9, zorder=1,
                              label='Stance Vector')

# Random Forest Classifier
rf_box = patches.Rectangle((0.58, 0.2), 0.16, 0.25, 
                          facecolor=colors['random_forest'][0], 
                          edgecolor=colors['random_forest'][1],
                          linewidth=2, alpha=0.9, zorder=1,
                          label='Random Forest')

# Add simple tree visualizations
tree_heights = [0.05, 0.07, 0.06, 0.08, 0.05]
tree_positions = [0.60, 0.63, 0.66, 0.69, 0.72]

for i, (pos, height) in enumerate(zip(tree_positions, tree_heights)):
    tree = patches.Rectangle((pos, 0.25), 0.02, height, 
                            facecolor='#ffb366', 
                            edgecolor=colors['random_forest'][1],
                            linewidth=1, alpha=0.9, zorder=2)
    ax.add_patch(tree)

# Dialog Act Classification
output_box = patches.Rectangle((0.82, 0.2), 0.15, 0.65, 
                              facecolor=colors['output'][0], 
                              edgecolor=colors['output'][1],
                              linewidth=2, alpha=0.9, zorder=1,
                              label='Classification')

# DAMSL Categories
damsl_box = patches.Rectangle((0.84, 0.25), 0.11, 0.55, 
                             facecolor='#f5b3b3', 
                             edgecolor=colors['output'][1],
                             linewidth=1, alpha=0.9, zorder=2)

# Add boxes to plot
for box in [input_box, audio_box, text_box, inter_box, vad_box, emotion_box, 
            certainty_box, engagement_box, stance_box, rf_box, output_box, damsl_box]:
    ax.add_patch(box)

# Add box labels
ax.text(0.125, 0.89, 'Feature Extraction', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.125, 0.84, 'Audio Features', ha='center', va='center', fontsize=10)
ax.text(0.125, 0.71, 'Textual Features', ha='center', va='center', fontsize=10)

ax.text(0.4, 0.89, 'Intermediate Representations', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.4, 0.8, 'VAD Estimation', ha='center', va='center', fontsize=10)
ax.text(0.4, 0.77, '(Valence, Arousal, Dominance)', ha='center', va='center', fontsize=8)
ax.text(0.4, 0.62, 'Categorical Emotion', ha='center', va='center', fontsize=10)
ax.text(0.4, 0.59, '(Ekman\'s 6 emotions)', ha='center', va='center', fontsize=8)
ax.text(0.4, 0.44, 'Certainty Estimation', ha='center', va='center', fontsize=10)
ax.text(0.4, 0.24, 'Engagement Estimation', ha='center', va='center', fontsize=10)

ax.text(0.66, 0.77, 'Stance Vector', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.66, 0.73, '(12 dimensions)', ha='center', va='center', fontsize=10)
ax.text(0.66, 0.68, '• Affective dimensions (3)', ha='center', va='center', fontsize=8)
ax.text(0.66, 0.65, '• Emotional states (6)', ha='center', va='center', fontsize=8)
ax.text(0.66, 0.62, '• Certainty dimension (1)', ha='center', va='center', fontsize=8)
ax.text(0.66, 0.59, '• Engagement dimensions (2)', ha='center', va='center', fontsize=8)

ax.text(0.66, 0.34, 'Random Forest', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.66, 0.31, 'Classifier', ha='center', va='center', fontsize=12, fontweight='bold')

ax.text(0.895, 0.77, 'Dialog Act', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.895, 0.73, 'Classification', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.895, 0.67, 'DAMSL Categories', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(0.895, 0.63, '• Statement-non-opinion (sd)', ha='center', va='center', fontsize=8)
ax.text(0.895, 0.60, '• Acknowledge/Backchannel (b)', ha='center', va='center', fontsize=8)
ax.text(0.895, 0.57, '• Statement-opinion (sv)', ha='center', va='center', fontsize=8)
ax.text(0.895, 0.54, '• Agree/Accept (aa)', ha='center', va='center', fontsize=8)
ax.text(0.895, 0.51, '• Abandoned/Turn-Exit (%)', ha='center', va='center', fontsize=8)
ax.text(0.895, 0.48, '• Yes-No-Question (qy)', ha='center', va='center', fontsize=8)
ax.text(0.895, 0.45, '• Non-verbal (x)', ha='center', va='center', fontsize=8)
ax.text(0.895, 0.42, '• Yes/No answers (ny/nn)', ha='center', va='center', fontsize=8)
ax.text(0.895, 0.39, '• Other acts...', ha='center', va='center', fontsize=8)
ax.text(0.895, 0.35, '43 dialogue act tags', ha='center', va='center', fontsize=9)

# Add arrows
# Audio Features to VAD
plt.arrow(0.18, 0.84, 0.13, 0, width=0.005, head_width=0.015, 
          head_length=0.01, fc='black', ec='black', zorder=3)

# Audio Features to Categorical Emotion
plt.arrow(0.18, 0.825, 0.10, -0.17, width=0.005, head_width=0.015, 
          head_length=0.01, fc='black', ec='black', zorder=3)

# Audio Features to Certainty
plt.arrow(0.18, 0.81, 0.09, -0.33, width=0.005, head_width=0.015, 
          head_length=0.01, fc='black', ec='black', zorder=3)

# Textual Features to VAD
plt.arrow(0.18, 0.735, 0.09, 0.04, width=0.005, head_width=0.015, 
          head_length=0.01, fc='black', ec='black', zorder=3)

# Textual Features to Categorical Emotion
plt.arrow(0.18, 0.71, 0.13, -0.07, width=0.005, head_width=0.015, 
          head_length=0.01, fc='black', ec='black', zorder=3)

# Textual Features to Certainty
plt.arrow(0.18, 0.685, 0.13, -0.29, width=0.005, head_width=0.015, 
          head_length=0.01, fc='black', ec='black', zorder=3)

# VAD to Stance Vector
plt.arrow(0.48, 0.80, 0.09, -0.05, width=0.005, head_width=0.015, 
          head_length=0.01, fc='black', ec='black', zorder=3)

# Categorical Emotion to Stance Vector
plt.arrow(0.48, 0.62, 0.09, 0.03, width=0.005, head_width=0.015, 
          head_length=0.01, fc='black', ec='black', zorder=3)

# Certainty to Stance Vector
plt.arrow(0.48, 0.44, 0.07, 0.15, width=0.005, head_width=0.015, 
          head_length=0.01, fc='black', ec='black', zorder=3)

# Engagement to Stance Vector
plt.arrow(0.48, 0.24, 0.06, 0.36, width=0.005, head_width=0.015, 
          head_length=0.01, fc='black', ec='black', zorder=3)

# Stance Vector to Random Forest
plt.arrow(0.66, 0.58, 0, -0.16, width=0.005, head_width=0.015, 
          head_length=0.01, fc='black', ec='black', zorder=3)

# Random Forest to Dialog Act Classification
plt.arrow(0.74, 0.33, 0.09, 0.17, width=0.005, head_width=0.015, 
          head_length=0.01, fc='black', ec='black', zorder=3)

# Direct Features Path (dashed)
verts = [(0.16, 0.67), (0.16, 0.45), (0.4, 0.35), (0.58, 0.32)]
codes = [Path.MOVETO] + [Path.CURVE4] * 3
path = Path(verts, codes)
direct_path = patches.PathPatch(path, facecolor='none', edgecolor='gray', 
                               linestyle='--', linewidth=1.5, zorder=3)
ax.add_patch(direct_path)
ax.text(0.3, 0.45, 'Direct feature paths', ha='center', va='center', 
        fontsize=8, fontstyle='italic', color='#555555')

# Add legend
legend_elements = [
    patches.Rectangle((0, 0), 1, 1, facecolor=colors['input'][0], edgecolor=colors['input'][1], label='Input Features'),
    patches.Rectangle((0, 0), 1, 1, facecolor=colors['intermediate'][0], edgecolor=colors['intermediate'][1], label='Intermediate Representations'),
    patches.Rectangle((0, 0), 1, 1, facecolor=colors['stance'][0], edgecolor=colors['stance'][1], label='Stance Vector'),
    patches.Rectangle((0, 0), 1, 1, facecolor=colors['random_forest'][0], edgecolor=colors['random_forest'][1], label='Random Forest'),
    patches.Rectangle((0, 0), 1, 1, facecolor=colors['output'][0], edgecolor=colors['output'][1], label='Dialog Act Classification')
]
ax.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9, 
          bbox_to_anchor=(0.5, 0.02))

# Add evaluation metrics at bottom
metrics_text = "Evaluation Metrics:   Accuracy   |   F1-Score   |   Confusion Matrix"
ax.text(0.5, 0.06, metrics_text, ha='center', va='center', fontsize=10, 
        bbox=dict(facecolor='#e1d5e7', edgecolor='#9673a6', boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.1)
plt.savefig('dialogue_act_classification_with_random_forest.png', dpi=300, bbox_inches='tight')
plt.show()