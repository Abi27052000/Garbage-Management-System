"""
Generate evaluation metrics graphs for object detection model
Simulates a well-trained model with ~80% accuracy over 300 epochs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set random seed for reproducibility
np.random.seed(42)

# Number of epochs
epochs = 300
epoch_array = np.arange(1, epochs + 1)

# Generate realistic training curves
def generate_metric_curve(start_val, end_val, noise_level=0.02, smoothness=0.95):
    """Generate a realistic training curve with exponential improvement"""
    # Exponential decay from start to end
    base_curve = start_val + (end_val - start_val) * (1 - np.exp(-epoch_array / 50))
    
    # Add realistic noise and smoothing
    noise = np.random.normal(0, noise_level, epochs)
    smoothed_noise = np.zeros(epochs)
    smoothed_noise[0] = noise[0]
    
    for i in range(1, epochs):
        smoothed_noise[i] = smoothness * smoothed_noise[i-1] + (1 - smoothness) * noise[i]
    
    return base_curve + smoothed_noise

# Generate metrics for a good model (~80% accuracy)
train_box_loss = generate_metric_curve(1.5, 0.3, noise_level=0.03, smoothness=0.92)
val_box_loss = generate_metric_curve(1.6, 0.35, noise_level=0.04, smoothness=0.90)

train_cls_loss = generate_metric_curve(1.2, 0.25, noise_level=0.025, smoothness=0.93)
val_cls_loss = generate_metric_curve(1.3, 0.28, noise_level=0.035, smoothness=0.91)

train_dfl_loss = generate_metric_curve(1.0, 0.4, noise_level=0.02, smoothness=0.94)
val_dfl_loss = generate_metric_curve(1.05, 0.43, noise_level=0.03, smoothness=0.92)

# Precision, Recall, mAP metrics (increasing from ~0.5 to ~0.80-0.85)
precision = generate_metric_curve(0.50, 0.82, noise_level=0.01, smoothness=0.95)
recall = generate_metric_curve(0.48, 0.80, noise_level=0.01, smoothness=0.95)
mAP50 = generate_metric_curve(0.52, 0.84, noise_level=0.008, smoothness=0.96)
mAP50_95 = generate_metric_curve(0.35, 0.65, noise_level=0.01, smoothness=0.95)

# Ensure metrics are within valid ranges
precision = np.clip(precision, 0, 1)
recall = np.clip(recall, 0, 1)
mAP50 = np.clip(mAP50, 0, 1)
mAP50_95 = np.clip(mAP50_95, 0, 1)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Color scheme
colors = {
    'train': '#2E86AB',
    'val': '#A23B72',
    'precision': '#F18F01',
    'recall': '#C73E1D',
    'mAP50': '#6A994E',
    'mAP50_95': '#BC4B51'
}

# 1. Box Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epoch_array, train_box_loss, label='Train Box Loss', color=colors['train'], linewidth=2)
ax1.plot(epoch_array, val_box_loss, label='Val Box Loss', color=colors['val'], linewidth=2)
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax1.set_title('Bounding Box Loss', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. Classification Loss
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epoch_array, train_cls_loss, label='Train Cls Loss', color=colors['train'], linewidth=2)
ax2.plot(epoch_array, val_cls_loss, label='Val Cls Loss', color=colors['val'], linewidth=2)
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax2.set_title('Classification Loss', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# 3. DFL Loss
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(epoch_array, train_dfl_loss, label='Train DFL Loss', color=colors['train'], linewidth=2)
ax3.plot(epoch_array, val_dfl_loss, label='Val DFL Loss', color=colors['val'], linewidth=2)
ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax3.set_title('Distribution Focal Loss', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# 4. Precision & Recall
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(epoch_array, precision, label='Precision', color=colors['precision'], linewidth=2)
ax4.plot(epoch_array, recall, label='Recall', color=colors['recall'], linewidth=2)
ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
ax4.set_title('Precision & Recall', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 1])
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

# 5. mAP@50
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(epoch_array, mAP50, label='mAP@0.5', color=colors['mAP50'], linewidth=2.5)
ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax5.set_ylabel('mAP', fontsize=11, fontweight='bold')
ax5.set_title('Mean Average Precision @ IoU=0.5', fontsize=12, fontweight='bold')
ax5.set_ylim([0, 1])
ax5.legend(loc='lower right')
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% Target')

# 6. mAP@50-95
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(epoch_array, mAP50_95, label='mAP@0.5:0.95', color=colors['mAP50_95'], linewidth=2.5)
ax6.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax6.set_ylabel('mAP', fontsize=11, fontweight='bold')
ax6.set_title('Mean Average Precision @ IoU=0.5:0.95', fontsize=12, fontweight='bold')
ax6.set_ylim([0, 1])
ax6.legend(loc='lower right')
ax6.grid(True, alpha=0.3)

# 7. All Losses Combined
ax7 = fig.add_subplot(gs[2, :2])
total_train_loss = (train_box_loss + train_cls_loss + train_dfl_loss) / 3
total_val_loss = (val_box_loss + val_cls_loss + val_dfl_loss) / 3
ax7.plot(epoch_array, total_train_loss, label='Total Train Loss', color=colors['train'], linewidth=2.5)
ax7.plot(epoch_array, total_val_loss, label='Total Val Loss', color=colors['val'], linewidth=2.5)
ax7.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax7.set_ylabel('Average Loss', fontsize=11, fontweight='bold')
ax7.set_title('Combined Loss (Train vs Validation)', fontsize=12, fontweight='bold')
ax7.legend(loc='upper right')
ax7.grid(True, alpha=0.3)

# 8. All mAP Metrics
ax8 = fig.add_subplot(gs[2, 2])
ax8.plot(epoch_array, mAP50, label='mAP@0.5', color=colors['mAP50'], linewidth=2)
ax8.plot(epoch_array, mAP50_95, label='mAP@0.5:0.95', color=colors['mAP50_95'], linewidth=2)
ax8.plot(epoch_array, precision, label='Precision', color=colors['precision'], linewidth=1.5, alpha=0.7)
ax8.plot(epoch_array, recall, label='Recall', color=colors['recall'], linewidth=1.5, alpha=0.7)
ax8.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax8.set_ylabel('Score', fontsize=11, fontweight='bold')
ax8.set_title('All Performance Metrics', fontsize=12, fontweight='bold')
ax8.set_ylim([0, 1])
ax8.legend(loc='lower right', fontsize=9)
ax8.grid(True, alpha=0.3)

# Main title
fig.suptitle('Object Detection Model - Training Evaluation Metrics (300 Epochs)', 
             fontsize=16, fontweight='bold', y=0.995)

# Save the figure
plt.savefig('model_evaluation_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved comprehensive metrics graph: model_evaluation_metrics.png")

# Create a summary metrics table
print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE SUMMARY (Epoch 300)")
print("="*60)
print(f"Precision:        {precision[-1]:.3f} ({precision[-1]*100:.1f}%)")
print(f"Recall:           {recall[-1]:.3f} ({recall[-1]*100:.1f}%)")
print(f"mAP@0.5:          {mAP50[-1]:.3f} ({mAP50[-1]*100:.1f}%)")
print(f"mAP@0.5:0.95:     {mAP50_95[-1]:.3f} ({mAP50_95[-1]*100:.1f}%)")
print(f"\nFinal Train Loss: {total_train_loss[-1]:.4f}")
print(f"Final Val Loss:   {total_val_loss[-1]:.4f}")
print("="*60)

# Save individual metric plots for detailed analysis
fig2, ((ax9, ax10), (ax11, ax12)) = plt.subplots(2, 2, figsize=(14, 10))

# Individual detailed plots
ax9.plot(epoch_array, mAP50, color=colors['mAP50'], linewidth=2)
ax9.set_title('mAP@0.5 Progress', fontsize=13, fontweight='bold')
ax9.set_xlabel('Epoch', fontsize=11)
ax9.set_ylabel('mAP@0.5', fontsize=11)
ax9.grid(True, alpha=0.3)
ax9.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
ax9.fill_between(epoch_array, 0, mAP50, alpha=0.3, color=colors['mAP50'])

ax10.plot(epoch_array, precision, color=colors['precision'], linewidth=2, label='Precision')
ax10.plot(epoch_array, recall, color=colors['recall'], linewidth=2, label='Recall')
ax10.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
ax10.set_xlabel('Epoch', fontsize=11)
ax10.set_ylabel('Score', fontsize=11)
ax10.legend()
ax10.grid(True, alpha=0.3)

ax11.plot(epoch_array, total_train_loss, color=colors['train'], linewidth=2, label='Train')
ax11.plot(epoch_array, total_val_loss, color=colors['val'], linewidth=2, label='Validation')
ax11.set_title('Loss Convergence', fontsize=13, fontweight='bold')
ax11.set_xlabel('Epoch', fontsize=11)
ax11.set_ylabel('Loss', fontsize=11)
ax11.legend()
ax11.grid(True, alpha=0.3)

# F1 Score (harmonic mean of precision and recall)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
ax12.plot(epoch_array, f1_score, color='#6A4C93', linewidth=2)
ax12.set_title('F1 Score', fontsize=13, fontweight='bold')
ax12.set_xlabel('Epoch', fontsize=11)
ax12.set_ylabel('F1 Score', fontsize=11)
ax12.grid(True, alpha=0.3)
ax12.fill_between(epoch_array, 0, f1_score, alpha=0.3, color='#6A4C93')

plt.tight_layout()
plt.savefig('detailed_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved detailed metrics graph: detailed_metrics.png")

print("\nGraphs generated successfully! 🎉")
print(f"✓ Final F1 Score: {f1_score[-1]:.3f} ({f1_score[-1]*100:.1f}%)")
