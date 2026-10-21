#!/usr/bin/env python3
"""Render the detector-choice table from the per-family F1 of the earlier seven-detector benchmark."""
from pathlib import Path
from itertools import combinations
import csv

HERE = Path(__file__).resolve().parents[1]
SRC = 'experiments/results/existing_metrics_long.csv'
rows = [r for r in csv.DictReader((HERE.parent / SRC).open())
        if r['benchmark'] == 'ARGENCON all families' and r['split'] == 'heldout']
f1, kind = {}, {}
for r in rows:
    f1.setdefault(r['family'], {})[r['model']] = float(r['f1'])
    kind[r['family']] = r['dga_type']
models = sorted({m for d in f1.values() for m in d})
assert len(f1) == 25 and len(models) == 7 and all(len(d) == 7 for d in f1.values())
groups = [sorted(f1), [f for f in f1 if kind[f] == 'WL'], [f for f in f1 if kind[f] == 'NW']]
assert [len(g) for g in groups] == [25, 4, 21]

def oracle(models, families):
    return sum(max(f1[f][m] for m in models) for f in families) / len(families)

CHOSEN = ('DomURLs-BERT', 'Logit', 'ModernBERT')
triples = sorted(combinations(models, 3), key=lambda s: -oracle(s, groups[0]))
assert len(triples) == 35 and triples[0] == CHOSEN

CITE = {'Logit': 'vranken2022detection', 'CNN': 'catania2019analysis', 'BiLSTM': 'namgung2022efficient',
        'Bilbo': 'highnam2021real', 'LA_Bin07': 'la2024llms', 'ModernBERT': 'warner2024modernbert',
        'DomURLs-BERT': 'mahdaouy2024domurls_bert'}

def name(m):
    m = m.replace('DomURLs-BERT', 'DomURLs\\_BERT').replace('LA_Bin07', 'LA\\_Bin07')
    return m

single_best = [max(oracle([m], g) for m in models) for g in groups]

def line(label, members, bold_best=False):
    cells = []
    for g, b in zip(groups, single_best):
        v = oracle(members, g)
        txt = format(v, '.4f')
        cells.append(r'\textbf{' + txt + '}' if bold_best and v == b else txt)
    return label + ' & ' + ' & '.join(cells) + r' \\'

body = [r'\grouprow{4}{Mean F1 of each detector}']
for m in sorted(models, key=lambda m: -oracle([m], groups[0])):
    row = line(name(m) + r'~\cite{' + CITE[m] + '}', [m], bold_best=True)
    body.append((r'\rowcolor{hl} ' if m in CHOSEN else '') + row)
body += [r'\midrule', r'\grouprow{4}{Oracle F1 of a set}',
         line('All seven', models),
         r'\rowcolor{hl} ' + line(r'Logit, ModernBERT, DomURLs\_BERT', CHOSEN)]
text = [r'\begin{table}[!t]',
        r'\caption{The seven detectors of our earlier benchmark on the 25 development families.}',
        r'\label{tab:detectors}', r'\centering\footnotesize', r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
        r' & \multicolumn{3}{c}{Development families} \\', r'\cmidrule(l){2-4}',
        r'Detector or set & All (25) & WL (4) & NW (21) \\', r'\midrule']
text += body + [r'\bottomrule', r'\end{tabular}',
        r'\par\smallskip\begin{minipage}{\linewidth}\scriptsize F1 comes from our earlier benchmark: it is computed for each family and averaged over the families, so it differs from the pooled F1 of Table~\ref{tab:test}. In parentheses: number of families. Oracle F1 takes, for each family, the F1 of the best detector in the set. Bold: best single detector in each column. Shaded: the three chosen detectors and their set, the best of the 35 sets of three.\end{minipage}',
        r'\end{table}']
(HERE / 'tables' / 'detector_choice.tex').write_text('\n'.join(text) + '\n')
print('\n'.join(text))
