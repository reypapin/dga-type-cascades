#!/usr/bin/env python3
"""Render the all-against-all table on F1, FPR, and time on the test set."""
from pathlib import Path
import csv

HERE = Path(__file__).resolve().parents[1]
RES = HERE.parent / 'extension' / 'results'
# (id in all_pairs, system in summary, row label, column label)
SYSTEMS = [
    ('Logit', 'AlwaysLogit', 'Logit', 'Lo'),
    ('ModernBERT', 'AlwaysModernBERT', 'ModernBERT', 'MB'),
    ('DomURLs_BERT', 'AlwaysDomURLs-BERT', 'DomURLs\\_BERT', 'DU'),
    ('C1', 'EnsembleMean', 'C1', 'C1'),
    ('C2', 'CascadeLogitDomURLs_t0.80', 'C2', 'C2'),
    ('C3', 'Diag_CascadeLogitModernBERT_t0.80', 'C3\\posthoc', 'C3'),
    ('C4', 'Router', 'C4', 'C4'),
    ('C5', 'C1_fpr_router', 'C5', 'C5'),
    ('C6', 'C2_type_cascade_rule', 'C6', 'C6'),
    ('C7', 'C3_type_cascade_dm', 'C7', 'C7'),
    ('C8', 'C3_type_cascade_jev', 'C8\\posthoc', 'C8'),
    ('C9', 'C4_judge_dm', 'C9', 'C9'),
    ('C10', 'C4_judge_jev', 'C10\\posthoc', 'C10'),
]
HIGHLIGHT = {'C3', 'C6'}

summary = {r['system']: r for r in csv.DictReader((RES / 'summary_final_test.csv').open(encoding='utf-8-sig'))}
f1 = {s[0]: float(summary[s[1]]['f1']) for s in SYSTEMS}
ms = {s[0]: float(summary[s[1]]['latency_mean_ms']) for s in SYSTEMS}
fpr = {s[0]: float(summary[s[1]]['fpr']) for s in SYSTEMS}


def compare(a, b):
    """+1 / -1 / 0 for a against b in F1, FPR, and time (values of Table III)."""
    sign = lambda x: (x > 0) - (x < 0)
    return sign(f1[a] - f1[b]), sign(fpr[b] - fpr[a]), sign(ms[b] - ms[a])


def outcome(a, b):
    c = compare(a, b)
    if min(c) >= 0 and max(c) > 0:
        return 'win'
    if max(c) <= 0 and min(c) < 0:
        return 'loss'
    return 'trade' if max(c) > 0 else 'tie'


ids = [s[0] for s in SYSTEMS]
wins = {a: sum(outcome(a, b) == 'win' for b in ids if b != a) for a in ids}
losses = {a: sum(outcome(a, b) == 'loss' for b in ids if b != a) for a in ids}
order = sorted(SYSTEMS, key=lambda s: (-wins[s[0]], losses[s[0]], -f1[s[0]]))

# Section VI-C states these results.
assert sum(wins.values()) == sum(losses.values())
assert {a for a in ids if losses[a] == 0} == {'Logit', 'C1', 'C2', 'C3', 'C5', 'C6', 'C8'}
assert {a for a in ids if wins[a] == max(wins.values())} == HIGHLIGHT and wins['C6'] == 5
assert outcome('C6', 'C3') == 'trade'

CELL = {
    'win': r'\cellcolor{winbg}\textcolor{passc}{$\blacktriangle$}',
    'loss': r'\cellcolor{lossbg}\textcolor{failc}{$\blacktriangledown$}',
    'trade': r'\textcolor{grouptext}{$\diamond$}',
    'tie': r'\textcolor{grouptext}{$\cdot$}',
}
assert not any(outcome(a, b) == 'tie' for a in ids for b in ids if a != b)
body = []
for a, _, label, _ in order:
    cells = [r'\cellcolor{diagbg}' if a == b else CELL[outcome(a, b)] for b, _, _, _ in order]
    head = (r'\cellcolor{hl}' if a in HIGHLIGHT else '') + label
    body.append(head + ' & ' + ' & '.join(cells) + f' & {wins[a]} & {losses[a]}' + r' \\')

ncol = len(order)
text = [r'\begin{table}[!t]',
        r'\caption{Pairwise comparison of the systems on F1, FPR, and time.}',
        r'\label{tab:pairs}', r'\centering\footnotesize', r'\setlength{\tabcolsep}{1.6pt}',
        r'\renewcommand{\arraystretch}{1.1}',
        r'\begin{tabular}{@{}l' + 'c' * ncol + r'@{\hspace{4pt}}cc@{}}', r'\toprule',
        r'Row vs.\ column & ' + ' & '.join(r'\scriptsize ' + s[3] for s in order) + r' & Won & Lost \\',
        r'\midrule'] + body + [r'\bottomrule', r'\end{tabular}',
        r'\par\smallskip\begin{minipage}{\linewidth}\scriptsize ' + CELL['win'].replace(r'\cellcolor{winbg}', '')
        + r': the row system beats the column system: it is at least as good in F1, FPR, and time, and better in at least one (Section~\ref{sec:selection}). '
        + CELL['loss'].replace(r'\cellcolor{lossbg}', '') + r': the column system beats the row system. ' + CELL['trade']
        + r': trade-off; each system is better in at least one metric. Values from Table~\ref{tab:test}. Rows and columns are sorted by wins. Won: number of systems that the row system beats; Lost: number of systems that beat it. Shaded: C3 and C6, which beat the most systems. Lo: Logit; MB: ModernBERT; DU: DomURLs\_BERT. \(\dagger\): post-hoc.\end{minipage}',
        r'\end{table}']
(HERE / 'tables' / 'pairwise_matrix.tex').write_text('\n'.join(text) + '\n')
for a in [s[0] for s in order]:
    print(f'{a:13s} won {wins[a]:2d} lost {losses[a]:2d}')
