#!/usr/bin/env python3
"""Render Table IV (all thirteen systems on the test set) from the frozen result CSVs."""
from pathlib import Path
import csv

HERE = Path(__file__).resolve().parents[1]
RES = HERE.parent / 'extension' / 'results'
ROWS = [  # (label in paper, id in all_pairs, system in summary, design, group)
    ('Logit', 'Logit', 'AlwaysLogit', 'single detector', 0),
    ('ModernBERT', 'ModernBERT', 'AlwaysModernBERT', 'single detector', 0),
    ('DomURLs\\_BERT', 'DomURLs_BERT', 'AlwaysDomURLs-BERT', 'single detector', 0),
    ('C1', 'C1', 'EnsembleMean', 'average of the three scores', 1),
    ('C2', 'C2', 'CascadeLogitDomURLs_t0.80', 'confidence cascade $\\rightarrow$ DomURLs\\_BERT', 1),
    ('C3\\posthoc', 'C3', 'Diag_CascadeLogitModernBERT_t0.80', 'confidence cascade $\\rightarrow$ ModernBERT', 1),
    ('C4', 'C4', 'Router', 'learned router', 2),
    ('C5', 'C5', 'C1_fpr_router', 'learned router, false-positive cost', 2),
    ('C6', 'C6', 'C2_type_cascade_rule', 'type cascade, dictionary', 3),
    ('C7', 'C7', 'C3_type_cascade_dm', 'type cascade, d1-3B', 3),
    ('C8\\posthoc', 'C8', 'C3_type_cascade_jev', 'type cascade, Jev', 3),
    ('C9', 'C9', 'C4_judge_dm', 'judge, d1-3B', 4),
    ('C10\\posthoc', 'C10', 'C4_judge_jev', 'judge, Jev', 4),
]
HIGHLIGHT = {'C3', 'C6'}

summary = {r['system']: r for r in csv.DictReader((RES / 'summary_final_test.csv').open(encoding='utf-8-sig'))}

# Rule of Section V-C: a beats b when it is at least as good in F1, FPR, and time, and better in one.
raw = {pid: (float(summary[s]['f1']), float(summary[s]['fpr']), float(summary[s]['latency_mean_ms'])) for _, pid, s, _, _ in ROWS}


def beats(a, b):
    (fa, pa, la), (fb, pb, lb) = raw[a], raw[b]
    return fa >= fb and pa <= pb and la <= lb and raw[a] != raw[b]


beaten = {b for b in raw if any(beats(a, b) for a in raw if a != b)}
wins = {a: sum(beats(a, b) for b in raw if b != a) for a in raw}
# Section VI-C names these systems.
assert beaten == {'ModernBERT', 'DomURLs_BERT', 'C4', 'C7', 'C9', 'C10'}, beaten
assert {a for a in raw if wins[a] == max(wins.values())} == HIGHLIGHT

# (column, decimals, scale, best is max?)
COLS = [('f1', 4, 1, True), ('fpr', 4, 1, False), ('p_1_100', 3, 1, True),
        ('wl_recall', 1, 100, True), ('nw_recall', 1, 100, True),
        ('new_family_recall', 1, 100, True), ('variant_of_seen_recall', 1, 100, True),
        ('latency_mean_ms', 3, 1, False)]
shown = {c: [round(float(summary[s][c]) * k, d) for _, _, s, _, _ in ROWS] for c, d, k, _ in COLS}
best = {c: (max if hi else min)(shown[c]) for c, _, _, hi in COLS}

body, last_group = [], 0
for (label, pid, system, design, group), i in zip(ROWS, range(len(ROWS))):
    r = summary[system]
    cells = [label, design]
    for c, d, k, _ in COLS:
        v = shown[c][i]
        txt = format(v, '.%df' % d)
        if v == best[c]:
            txt = r'\textbf{' + txt + '}'
        if c in ('f1', 'fpr') and int(r['n_seeds']) > 1:
            txt += r'\,{\scriptsize$\pm$' + format(float(r[c + '_sd']), '.4f') + '}'
        cells.append(txt)
    if pid in beaten:
        cells = [r'\textcolor{dimtext}{' + c + '}' for c in cells]
    line = ' & '.join(cells) + r' \\'
    if pid in HIGHLIGHT:
        line = r'\rowcolor{hl} ' + line
    if group != last_group:
        body.append(r'\midrule')
        last_group = group
    body.append(line)

text = [r'\begin{table*}[!t]',
        r'\caption{Results of the thirteen systems on the test set at their default thresholds.}',
        r'\label{tab:test}', r'\centering\footnotesize', r'\setlength{\tabcolsep}{5.8pt}',
        r'\begin{tabular}{llcccccccc}', r'\toprule',
        r' & & \multicolumn{3}{c}{Detection} & \multicolumn{4}{c}{Recall (\%)} & \multicolumn{1}{c}{Cost} \\',
        r'\cmidrule(lr){3-5}\cmidrule(lr){6-9}\cmidrule(l){10-10}',
        r'System & Design & F1 & FPR & $P_{1:100}$ & WL & NW & New & Var. & ms \\',
        r'\midrule'] + body + [r'\bottomrule', r'\end{tabular}',
        r'\par\smallskip\begin{minipage}{\linewidth}\scriptsize C4 and C5: mean over five routers, with the sample standard deviation of F1 and FPR. $P_{1:100}$: precision at one DGA domain per 100 benign domains. Recall in percent on the WL and NW domains, on the 18 new families (New), and on the 9 variants of seen families (Var.). ms: mean time per domain, with decision-model calls but without router and dictionary overhead (Section~\ref{sec:limitations}); C8 and C10 times include internet round trips. Bold value: best in its column. Gray: beaten by another system (Section~\ref{sec:selection}). Shaded: C3 and C6, which beat the most systems. \(\dagger\): post-hoc.\end{minipage}',
        r'\end{table*}']
(HERE / 'tables' / 'test_summary.tex').write_text('\n'.join(text) + '\n')
print('\n'.join(body))
