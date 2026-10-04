"""Training progress plots. Never evaluates a model or accesses test data."""
from pathlib import Path
import csv
import math


def plot_training_history(history_path, output_path=None):
    import matplotlib.pyplot as plt
    path = Path(history_path)
    if not path.is_file() or not path.stat().st_size:
        return None
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    def value(row, key):
        try:
            return float(row.get(key, ''))
        except (ValueError, TypeError):
            return math.nan
    # Ignore incomplete lines while inspecting a running training process.
    rows = [r for r in rows if math.isfinite(value(r, 'epoch')) and math.isfinite(value(r, 'train_loss'))]
    if not rows:
        return None
    epochs = [value(r, 'epoch') for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Training evolution', fontsize=15)
    def curve(ax, key, label, color, divisor=1):
        vals = [value(r, key)/divisor for r in rows]
        if any(math.isfinite(v) for v in vals):
            ax.plot(epochs, vals, marker='o', markersize=3, label=label, color=color)
    for key, label, color in [('train_loss','Total loss','#2563eb'),
                              ('train_bce_loss','BCE (unweighted)','#e67e22'),
                              ('train_top1_loss','Top-1 loss (unweighted)','#00876c')]:
        curve(axes[0,0], key, label, color)
    axes[0,0].set(title='Training loss', ylabel='Loss')
    curve(axes[0,1], 'val_mean_makespan', 'RCPSP validation', '#2563eb')
    valid = [(e, value(r,'val_mean_makespan')) for e,r in zip(epochs,rows)
             if math.isfinite(value(r,'val_mean_makespan'))]
    if valid:
        best_epoch, best = min(valid, key=lambda pair: pair[1])
        running = []
        minimum = math.inf
        for _, v in valid:
            minimum = min(minimum,v)
            running.append(minimum)
        axes[0,1].step([e for e,_ in valid], running, where='post', linestyle='--',
                       color='#00876c', label='Best so far')
        axes[0,1].scatter([best_epoch],[best], marker='*', s=150, color='#b45309', zorder=5,
                          label=f'Best: epoch {best_epoch:g} · {best:.4f}')
    else:
        axes[0,1].text(.5,.5,'Validation pending', transform=axes[0,1].transAxes, ha='center')
    axes[0,1].set(title='Mean validation makespan (lower is better)', ylabel='Instance time units')
    for key,label,color in [('train_top1_any_positive','Top-1','#2563eb'),
                             ('train_binary_accuracy','Accuracy','#e67e22'),
                             ('train_f1','F1','#00876c')]:
        curve(axes[1,0],key,label,color)
    axes[1,0].set(title='Training metrics (not validation/test)', ylabel='Proportion', ylim=(0,1.02))
    curve(axes[1,1],'seconds','Complete epoch','#2563eb',60)
    curve(axes[1,1],'val_seconds','Validation (included in total)','#e67e22',60)
    axes[1,1].set(title='Computation time per epoch', ylabel='Minutes')
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
        if len(epochs)==1:
            ax.set_xlim(epochs[0]-.5,epochs[0]+.5)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(destination,dpi=160)
    return fig
