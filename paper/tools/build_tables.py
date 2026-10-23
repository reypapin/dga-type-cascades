#!/usr/bin/env python3
"""Render supplied result CSVs into manuscript tables; never runs inference."""
from pathlib import Path
import csv
import re
from statistics import mean

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / 'tables'
OUT.mkdir(exist_ok=True)
LEDGER = []
NAMES = [('Logit','AlwaysLogit'),('ModernBERT','AlwaysModernBERT'),
 ('DomURLs\\_BERT','AlwaysDomURLs-BERT'),('C1','EnsembleMean'),
 ('C2','CascadeLogitDomURLs_t0.80'),('C3\\posthoc','Diag_CascadeLogitModernBERT_t0.80'),
 ('C4','Router'),('C5','C1_fpr_router'),('C6','C2_type_cascade_rule'),
 ('C7','C3_type_cascade_dm'),('C8\\posthoc','C3_type_cascade_jev'),
 ('C9','C4_judge_dm'),('C10\\posthoc','C4_judge_jev')]
IDS = ['Logit','ModernBERT','DomURLs_BERT'] + ['C'+str(i) for i in range(1,11)]

def read(name):
    return list(csv.DictReader((ROOT / name).open()))

def tex(s):
    return str(s).replace('_',r'\_')

def fmt(x, n, where, source, row, pct=False, sign=False):
    x = float(x) * (100 if pct else 1)
    if abs(x) < 0.5 * 10 ** (-n):
        x = 0.0
    val = format(x, ('+' if sign else '') + '.' + str(n) + 'f')
    LEDGER.append((val + (' percent' if pct else ''), where, source + '; ' + row))
    return val

# Tables kept out of the paper; they go to the public repository with the CSVs.
# Table IV of the paper is drawn by tools/build_test_summary.py; this plain version goes to the repository.
SUPPLEMENTARY = {'test_summary_plain.tex', 'development.tex', 'random_intervals.tex', 'full_families.tex', 'full_matrix.tex',
                 'records.tex', 'decision_trial.tex', 'family_composition_full.tex'}

def put(name, text):
    assert len(text.splitlines()) <= 150
    folder = HERE / 'supplementary' if name in SUPPLEMENTARY else OUT
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(text)

def table(name, caption, label, spec, header, rows, wide=False, note=''):
    env = 'table*' if wide else 'table'
    text = [r'\begin{' + env + r'}[!t]',r'\caption{' + caption + '}',
      r'\label{' + label + '}',(r'\centering\scriptsize' if name in ('full_families.tex', 'full_matrix.tex') else r'\centering\footnotesize'),
      r'\setlength{\tabcolsep}{' + ('2.4pt' if name == 'fixed_fpr.tex' else '3.2pt') + '}',r'\begin{tabular}{' + spec + '}',
      r'\toprule',header + r' \\',r'\midrule'] + [r + r' \\' for r in rows]
    text += [r'\bottomrule',r'\end{tabular}']
    if note:
        text += [r'\par\smallskip\begin{minipage}{\linewidth}\scriptsize ' + note + r'\end{minipage}']
    text += [r'\end{' + env + '}']
    put(name, '\n'.join(text)+'\n')

pairs_src = 'extension/results/test_v2_all_pairs.csv'
pairs = read(pairs_src)
router_pairs_src = 'extension/results/revision_router_seed_intervals.csv'
router_pairs = {(r['a'], r['b']): r for r in read(router_pairs_src) if r['metric'] == 'f1'}
pairs = [router_pairs.get((r['a'], r['b']), r) for r in pairs]
pair_sources = pairs_src + '; router comparisons: ' + router_pairs_src
RECORDS = {}
lookup = {}
for r in pairs:
    a,b = r['a'],r['b']
    lookup[a,b] = [float(r[k]) for k in ('diff','low','high')]
    lookup[b,a] = [-float(r['diff']),-float(r['high']),-float(r['low'])]
records = []
for name in IDS:
    wins = sum(lookup[name,b][1] > 0 for b in IDS if b != name)
    losses = sum(lookup[name,b][2] < 0 for b in IDS if b != name)
    label = next(t[0] for t in NAMES if IDS[NAMES.index(t)] == name)
    RECORDS[name] = (wins, losses)
    records.append(label+' & '+str(wins)+' & '+str(losses))
    LEDGER.append((str(wins)+' wins; '+str(losses)+' losses','Table records: '+label,pair_sources+'; incident pairs for '+name))
for split in ['final_test','gate1']:
    src = 'extension/results/summary_' + split + '.csv'
    rows = {r['system']:r for r in read(src)}
    body = []
    for label, key in NAMES:
        r = rows[key]
        loc = 'Table ' + ('test' if split == 'final_test' else 'development') + ': ' + label
        def f(k,n=4,pct=False):
            return fmt(r[k],n,loc,src,'system='+key+'; column='+k,pct)
        f1, fpr = f('f1'), f('fpr')
        if int(r['n_seeds']) > 1:
            f1 = '$' + f1 + r'\pm' + f('f1_sd') + '$'
            fpr = '$' + fpr + r'\pm' + f('fpr_sd') + '$'
        cells = [label,f1,fpr,f('p_1_100',3),f('wl_recall',1,True),f('nw_recall',1,True)]
        if split == 'final_test':
            cells += [f('new_family_recall',1,True),f('variant_of_seen_recall',1,True)]
        cells += [f('latency_mean_ms',3)]
        if split == 'final_test':
            cells += [str(x) for x in RECORDS[IDS[NAMES.index((label, key))]]]
        # Group the rows: single models, C1-C3, routers, type cascades, judges.
        body.append((r'\midrule ' if key in ('EnsembleMean','Router','C2_type_cascade_rule','C4_judge_dm') else '') + ' & '.join(cells))
    if split == 'final_test':
        table('test_summary_plain.tex','Results of the thirteen systems on the test set.',
          'tab:test','@{}lcccccccccc@{}',
          'System & F1 & FPR & $P_{1:100}$ & WL & NW & New & Variant & ms & W & L',body,True,
          r'C4 and C5: mean over five routers, $\pm$ sample standard deviation for F1 and FPR. WL, NW, New, Variant: recall in percent on these groups. $P_{1:100}$: precision at one DGA domain per 100 benign domains. ms: recorded mean time per domain, with decision-model calls but without router and dictionary overhead (Section~\ref{sec:limitations}); C8 and C10 times include internet round trips. W, L: clear F1 wins and losses in the 78 pairwise comparisons. \(\dagger\): post-hoc.')
    else:
        table('development.tex','Development results. Slice columns are recall in percent. C4/C5 report means and sample standard deviations for F1 and FPR over five seeds; other entries are means. C5, C6, C7, and C9 were frozen before the test run; C3, C8, and C10 are post-hoc.',
          'tab:development','@{}lrrrrrr@{}','System & F1 & FPR & $P_{1:100}$ & WL & NW & ms',body,True,
          r'\(\dagger\): post-hoc. Jev times are internet round trips. Threshold provenance is given in Table~\ref{tab:selection}.')

src = pair_sources
table('records.tex','Descriptive pairwise F1 records among the thirteen systems. A win or loss means that the unadjusted interval excludes zero. The counts are not a global ranking.',
 'tab:records','@{}lrr@{}','System & Clear wins & Clear losses',records)
keyrows=[]
for other in ['ModernBERT','C1','C2','C3','C4','C7','C8','C10']:
    vals=lookup['C6',other]
    cells=[r'C6 $-$ '+other+(r'\posthoc' if other in ('C3','C8','C10') else '')]
    cells += [fmt(vals[0],4,'Table key pairs',src,'C6 minus '+other,sign=True)]
    cells += ['['+fmt(vals[1],3,'Table key pairs',src,'C6 minus '+other+' low',sign=True)+', '+fmt(vals[2],3,'Table key pairs',src,'C6 minus '+other+' high',sign=True)+']']
    cells += ['yes' if vals[1]>0 or vals[2]<0 else 'no']
    keyrows.append(' & '.join(c.replace('-0.', '$-$0.') for c in cells))
table('key_pairs.tex','F1 differences between C6 and selected systems on the test set.',
 'tab:key-pairs','@{}lccc@{}','Pair & Difference & 95\\% interval & Clear',keyrows,note=r'Paired bootstrap over DGA families and benign domains, not adjusted for multiple comparisons. Clear: the interval excludes zero. \(\dagger\): post-hoc.')

src = 'extension/results/test_v2_fixed_fpr.csv'
fixed = read(src)
c4src = 'extension/results/test_v2_fixed_fpr_c4.csv'
c4fixed = read(c4src)
fixedrows = []
for label, key in [NAMES[0],NAMES[1],NAMES[2],NAMES[3],NAMES[6],NAMES[7]]:
    cells = [label + (r'\textsuperscript{a}' if key=='Router' else '')]
    for target in (0.001,0.01):
        fsrc = c4src if key=='Router' else src
        data = c4fixed if key=='Router' else fixed
        if key in ('Router','C1_fpr_router'):
            prefix = 'C4_s' if key=='Router' else 'C1_s'
            chosen = [r for r in data if r['system'].startswith(prefix) and float(r['target_fpr'])==target]
        else:
            chosen = [r for r in data if r['system']==key and float(r['target_fpr'])==target]
        for col,n,pct in [('recall',3,False),('fpr',2,True),('p_1_100',3,False)]:
            value = mean(float(r[col]) for r in chosen)
            cells += [fmt(value,n,'Table fixed thresholds',fsrc,'system='+key+'; target='+str(target)+'; mean '+col,pct)+ (r'\%' if pct else '')]
    fixedrows.append(' & '.join(cells))
table('fixed_fpr.tex','Test results at thresholds chosen on the development set for FPR targets of 0.1\\% and 1\\%.',
 'tab:fixed-fpr','@{}lcccccc@{}',' & \\multicolumn{3}{c}{0.1\\% target} & \\multicolumn{3}{c}{1\\% target} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7} System & Recall & FPR & $P_{1:100}$ & Recall & FPR & $P_{1:100}$',fixedrows,False,
 r'FPR: achieved test FPR in percent. C4 and C5: mean over five routers, each with its own threshold. \textsuperscript{a}Later descriptive analysis; the other rows were specified before the test run.')

src = 'extension/results/test_v2_per_family.csv'
families = read(src)
columns = ['Logit','ModernBERT','DomURLs-BERT','Ensemble','CascadeCACIC','Diag_LogitModernBERT',
 'RouterCACIC','C1','C2','C3','C3_jev','C4','C4_jev']
order = ['nymaim2','gozi_luther','gozi_nasa','gozi_gpl','beebone','volatilecedar','diamondfox',
 'tofsee','xxhex','goznym','mirai','pykspa2s','pykspa2','tempedrevetdd','pushdotid','ekforward',
 'vidrotid','bobax','shifu','madmax','urlzone','makloader','sutra','blackhole','ud2','ebury','gameover_p2p']
fmap = {r['family']:r for r in families}
familyrows = []
for fam in order:
    r = fmap[fam]
    cells = [tex(fam),r['dga_type'],'N' if r['tier']=='new_family' else 'V',r['n']]
    LEDGER.append((r['n'],'Table full families: '+fam,src+'; family='+fam+'; n'))
    cells += [fmt(r[c],1,'Table full families: '+fam,src,'family='+fam+'; '+c,True) for c in columns]
    familyrows.append(' & '.join(cells))
table('full_families.tex','Test recall in percent for every DGA family. N denotes a new family group and V a declared variant of a training family. C4/C5 average five seeds.',
 'tab:full-families','@{}lccr'+'r'*13+'@{}',
 'Family & Type & Tier & $n$ & Logit & Modern & DomURLs & C1 & C2 & C3\\posthoc & C4 & C5 & C6 & C7 & C8\\posthoc & C9 & C10\\posthoc',
 familyrows,True,r'No benign rows are included in these recalls. Benign false-positive rates are reported in Table~\ref{tab:test}. Modern and DomURLs abbreviate ModernBERT and DomURLs\_BERT in this table only.')
selected=[]
for fam in order[:9]:
    r=fmap[fam]
    cells=[tex(fam)]
    cells += [fmt(r[c],1,'Table selected families: '+fam,src,'family='+fam+'; '+c,True) for c in ('Logit','ModernBERT','DomURLs-BERT','C2','C4','C4_jev')]
    selected.append(' & '.join(cells))
table('selected_families.tex','Recall (\\%) on the four WL groups and the hardest NW groups of the test set.',
 'tab:selected-families','@{}lcccccc@{}','Family & Logit & Modern & DomURLs & C6 & C9 & C10\\posthoc',selected,note=r'Modern: ModernBERT. DomURLs: DomURLs\_BERT. C9 and C10 are judges. \(\dagger\): post-hoc.')
matrixrows=[]
for a in IDS:
    cells=[tex(a)+(r'\posthoc' if a in ('C3','C8','C10') else '')]
    for b in IDS:
        if a==b:
            cells.append('--')
        else:
            d,l,h=lookup[a,b]
            v=fmt(d,3,'Table full matrix', pair_sources,a+' minus '+b,sign=True)
            cells.append('$'+v+(r'^{*}' if l>0 or h<0 else '')+'$')
    matrixrows.append(' & '.join(cells))
table('full_matrix.tex','Full descriptive pairwise F1 matrix. Each cell is row minus column. An asterisk marks an unadjusted bootstrap interval that excludes zero. A displayed zero can hide a small nonzero difference.',
 'tab:matrix','@{}l'+'r'*13+'@{}','Row $-$ column & '+' & '.join(tex(x)+(r'\posthoc' if x in ('C3','C8','C10') else '') for x in IDS),
 matrixrows,True,r'C4/C5 comparisons draw one of the five router seeds in each replicate. There is no adjustment across the 78 comparisons. The matrix does not establish equivalence.')
# Keep all thirteen matrix columns within IEEE full text width.
matrix_path = HERE / 'supplementary' / 'full_matrix.tex'
matrix_path.write_text(matrix_path.read_text().replace(
    r'{\tabcolsep}{3.2pt}', r'{\tabcolsep}{2pt}'))
src='extension/results/test_v2_bootstrap.csv'
boot=read(src)
controlrows=[]
for old,label in [('C2','C6'),('C3','C7')]:
    for r in [r for r in boot if r['a']==old and '_control_s' in r['b']]:
        row=label+' minus random seed '+r['b'].split('_s')[-1]
        vals=[fmt(r[k],3,'Table random intervals',src,'a='+r['a']+'; b='+r['b']+'; '+k,sign=True) for k in ('diff','low','high')]
        controlrows.append(label+' & '+r['b'].split('_s')[-1]+' & '+vals[0]+' & ['+vals[1]+', '+vals[2]+']')
table('random_intervals.tex','Preregistered F1 comparisons against each route-matched random-control seed. Every interval excludes zero in favor of the configuration.',
 'tab:random-intervals','@{}lrrr@{}','System & Seed & Difference & 95\\% interval',controlrows)

# The trial timings retain their source precision; full-run means are separate.
trial_src = 'extension/results/smoke_decision_models.csv'
trial = read(trial_src)
trial_rows = []
for key, label, order, flip in [
        ('d1', 'd1-3B', 'random_option_first', '5.0'),
        ('laya', 'Laya', 'random_option_first', '35.0'),
        ('julia', 'Julia-1', 'wl_option_first', '1.5')]:
    r = next(r for r in trial if r['model'] == key and r['order'] == order)
    cells = [label]
    for col in ('auc_wl_vs_nw', 'auc_dga_vs_benign'):
        cells.append(fmt(r[col], 3, 'Table decision trial', trial_src,
                         'model=' + key + '; order=' + order + '; ' + col))
    cells += [flip + r'\%', r['latency_ms']]
    LEDGER.append((r['latency_ms'] + ' ms', 'Table decision trial: ' + label,
                   trial_src + '; model=' + key + '; order=' + order + '; latency_ms'))
    trial_rows.append(' & '.join(cells))
trial_rows += [r'Jev (API) & 0.992 & 0.907 & 4.0\% & 487',
               r'Dictionary & 0.982 & --- & --- & ---']
LEDGER.append(('487 ms', 'Table decision trial: Jev',
               'analisis/resultados_extension_q2.html; Decision models table, Jev row'))
table('decision_trial.tex',
      'Decision-model trial on development domains. AUC values use the option orders specified below. Flip is the percentage of WL answers that change when the options are reversed. Jev is a later API reference.',
      'tab:decision-trial', '@{}lrrrr@{}',
      'Model & WL AUC & DGA AUC & Flip & ms/call', trial_rows,
      note=r'The local chosen order places random-looking/benign first for d1-3B and Laya. The displayed Julia-1 row places real words/DGA first. Jev timing is an internet round trip. Dictionary timing is excluded here; see Section~\ref{sec:latency}.')

# Family composition comes from the released taxonomy CSVs.
taxonomy = read('data/family_taxonomy.csv')
test_taxonomy = [r for r in read('data/final_test_family_taxonomy.csv') if r['included'] == '1']
def family_name(r, kind_key, variant=False):
    name = tex(r['family']) + (r'$^{\mathrm{w}}$' if r[kind_key] == 'WL' else '')
    return name + (' (' + tex(r['base_family']) + ')' if variant else '')
family_groups = [
    ('Training (54)', sorted((r for r in taxonomy if r['split'] == 'training'), key=lambda r:r['family']), 'type', False),
    ('Development (25)', sorted((r for r in taxonomy if r['split'] == 'heldout'), key=lambda r:r['family']), 'type', False),
    ('Test, new (18)', sorted((r for r in test_taxonomy if r['tier'] == 'new_family'), key=lambda r:r['family']), 'dga_type', False),
    ('Test, variants (9)', sorted((r for r in test_taxonomy if r['tier'] != 'new_family'), key=lambda r:r['family']), 'dga_type', True),
]
assert [len(group) for _,group,_,_ in family_groups] == [54,25,18,9]
family_rows = [label+' & '+', '.join(family_name(r,key,variant) for r in group)+r' \\' for label,group,key,variant in family_groups]
family_body = ('\n'+r'\midrule'+'\n').join(family_rows)
family_caption = 'Family composition. Superscript w marks WL groups; all other groups are NW. Parentheses identify the training family of a declared variant. A new group is not a source-declared variant.'
put('family_composition_full.tex','\n'.join([r'\begin{table*}[!t]',r'\caption{'+family_caption+'}',r'\label{tab:families-full}',r'\centering\scriptsize',r'\begin{tabular}{@{}p{0.15\textwidth}p{0.80\textwidth}@{}}',r'\toprule',family_body,r'\bottomrule',r'\end{tabular}',r'\end{table*}'])+'\n')
testrows=[family_rows[2].replace('Test, new (18)','New (18)'),family_rows[3].replace('Test, variants (9)','Variants (9)')]
put('family_composition.tex','\n'.join([r'\begin{table}[!t]',r'\caption{The 27 test family groups.}',r'\label{tab:families}',r'\centering\footnotesize',r'\begin{tabular}{@{}p{0.16\columnwidth}p{0.79\columnwidth}@{}}',r'\toprule',testrows[0],r'\midrule',testrows[1],r'\bottomrule',r'\end{tabular}',r'\par\smallskip\begin{minipage}{\linewidth}\scriptsize $^{\mathrm{w}}$: WL group; all other groups are NW. In parentheses: the training family of a declared variant.\end{minipage}',r'\end{table}'])+'\n')
LEDGER.append(('54; 25; 18; 9; 27; WL labels','Table family composition','data/family_taxonomy.csv; data/final_test_family_taxonomy.csv; included groups'))
lines=['# Numerical provenance','','Display | Location in the paper | Source file and row','--- | --- | ---']
lines += [' | '.join(x) for x in LEDGER]
extra=HERE/'NUMBERS_PROSE.md'
if extra.exists():
    lines += ['','## Prose, definitions, and selection details','']+extra.read_text().splitlines()
(HERE/'NUMBERS.md').write_text('\n'.join(lines)+'\n')
print('Rendered',len(list(OUT.glob('*.tex'))),'tables; ledger rows:',len(LEDGER))
