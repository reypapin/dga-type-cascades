#!/usr/bin/env python3
"""Dot plot of the 13 systems on F1, FPR, and time (test set), for the GitHub README.

Reads extension/results/summary_final_test.csv and applies the rule of Section V-C:
A beats B when A is at least as good in F1, FPR, and time, and better in at least one.
Writes assets/three_metrics_light.png and assets/three_metrics_dark.png.
Usage: python3 tools/plot_three_metrics.py RESULTS_CSV OUT_DIR
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D

NAMES = {
    'AlwaysLogit': 'Logit', 'AlwaysModernBERT': 'ModernBERT', 'AlwaysDomURLs-BERT': 'DomURLs_BERT',
    'EnsembleMean': 'C1', 'CascadeLogitDomURLs_t0.80': 'C2', 'Diag_CascadeLogitModernBERT_t0.80': 'C3',
    'Router': 'C4', 'C1_fpr_router': 'C5', 'C2_type_cascade_rule': 'C6', 'C3_type_cascade_dm': 'C7',
    'C3_type_cascade_jev': 'C8', 'C4_judge_dm': 'C9', 'C4_judge_jev': 'C10',
}
ORDER = ['Logit', 'ModernBERT', 'DomURLs_BERT', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
POSTHOC = {'C3', 'C8', 'C10'}

THEMES = {
    'light': dict(ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
                  c3='#2a78d6', c6='#eb6834', other='#52514e', beaten='#b5b3ab'),
    'dark': dict(ink='#f0efec', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835',
                 c3='#3987e5', c6='#d95926', other='#c3c2b7', beaten='#6a6964'),
}


def load(path: Path) -> dict:
    rows = {NAMES[r['system']]: r for r in csv.DictReader(path.open(encoding='utf-8-sig'))}
    return {n: (float(rows[n]['f1']), 100 * float(rows[n]['fpr']), float(rows[n]['latency_mean_ms'])) for n in ORDER}


def beats(a, b) -> bool:
    ge = a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2]
    return ge and (a[0] > b[0] or a[1] < b[1] or a[2] < b[2])


def plot(data: dict, theme: str, out: Path) -> None:
    t = THEMES[theme]
    wins = {n: sum(beats(data[n], data[m]) for m in ORDER if m != n) for n in ORDER}
    beaten = {n for n in ORDER if any(beats(data[m], data[n]) for m in ORDER if m != n)}
    top = {n for n in ORDER if wins[n] == max(wins.values())}
    assert top == {'C3', 'C6'} and beaten == {'ModernBERT', 'DomURLs_BERT', 'C4', 'C7', 'C9', 'C10'}

    plt.rcParams.update({'font.family': 'Inter', 'font.size': 11, 'text.color': t['ink'],
                         'axes.labelcolor': t['ink2'], 'xtick.color': t['muted'], 'ytick.color': t['ink']})
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.6), sharey=True, gridspec_kw={'wspace': 0.08})
    fig.patch.set_alpha(0)
    ys = list(range(len(ORDER)))[::-1]
    panels = [
        (0, 'F1', 'higher is better', (0.855, 0.938), False, lambda v: f'{v:.3f}'),
        (1, 'False-positive rate (%)', 'lower is better', (1.6, 10.9), False, lambda v: f'{v:.2f}'),
        (2, 'Time per domain (ms, log scale)', 'lower is better', (0.04, 400), True, lambda v: f'{v:.2f}'),
    ]
    for ax, (k, title, hint, lim, log, fmt) in zip(axes, panels):
        ax.set_facecolor('none')
        if log:
            ax.set_xscale('log')
            ax.set_xticks([0.1, 1, 10, 100], ['0.1', '1', '10', '100'])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(*lim)
        for y in ys:
            ax.axhline(y, color=t['grid'], lw=0.8, zorder=0)
        for y, n in zip(ys, ORDER):
            v = data[n][k]
            if n in top:
                color = t['c3'] if n == 'C3' else t['c6']
                ax.scatter(v, y, s=95, color=color, edgecolor=t['ink'] if theme == 'dark' else 'white', lw=1.2, zorder=3)
                ax.annotate(fmt(v), (v, y), xytext=(0, 9), textcoords='offset points', ha='center',
                            fontsize=9, color=t['ink'], fontweight='semibold')
            elif n in beaten:
                ax.scatter(v, y, s=60, facecolor='none', edgecolor=t['beaten'], lw=1.5, zorder=2)
            else:
                ax.scatter(v, y, s=60, color=t['other'], zorder=2)
        ax.set_title(title, loc='left', fontsize=11.5, fontweight='semibold', color=t['ink'], pad=20)
        ax.text(0, 1.015, hint, transform=ax.transAxes, fontsize=9.5, color=t['muted'], va='bottom')
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(t['axis'])
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=9.5, color=t['axis'])
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([n + ('$^{\\dagger}$' if n in POSTHOC else '') for n in ORDER])
    for label, n in zip(axes[0].get_yticklabels(), ORDER):
        if n in top:
            label.set_fontweight('bold')
        elif n in beaten:
            label.set_color(t['muted'])
    axes[0].set_ylim(-0.7, len(ORDER) - 0.3)
    handles = [
        Line2D([], [], ls='', marker='o', ms=9, color=t['c3'], label='C3$^{\\dagger}$'),
        Line2D([], [], ls='', marker='o', ms=9, color=t['c6'], label='C6'),
        Line2D([], [], ls='', marker='o', ms=7.5, color=t['other'], label='beaten by no system'),
        Line2D([], [], ls='', marker='o', ms=7.5, mfc='none', mec=t['beaten'], mew=1.5, label='beaten by another system'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.53, -0.015), labelcolor=t['ink2'], handletextpad=0.3, columnspacing=1.6)
    fig.text(0.125, -0.06, 'Test set: 27 family groups unseen in training and 50,000 benign names. '
             'A beats B when it is at least as good on all three metrics and better on one. † post-hoc.',
             fontsize=9, color=t['muted'])
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.15, transparent=True)
    plt.close(fig)


if __name__ == '__main__':
    data = load(Path(sys.argv[1]))
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        plot(data, theme, out_dir / f'three_metrics_{theme}.png')
    print('written:', *(f'three_metrics_{t}.png' for t in THEMES))
