from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parent;out=root/'results';s=pd.read_csv(out/'classical_performance.csv');main=pd.read_csv(root.parent/'training/results/performance_summary.csv');
extra=[]
for name in ['chen_performance.csv','pugraph_performance.csv']:
 p=out/name
 if p.exists() and p.stat().st_size>1:
  try:extra.append(pd.read_csv(p))
  except pd.errors.EmptyDataError:pass
if extra:s=pd.concat([s,*extra],ignore_index=True)
wide=[]
for t in ['Tg','FFV','Tc','Density','Rg']:
 r={'Property':t}
 for b,label in [('full','Poly-Prophet recomputed'),('mean','Mean predictor'),('morgan_ridge','Morgan-Ridge'),('morgan_rf','Morgan-RF'),('chen_rnn','Chen 2021 RNN (transferred)'),('pugraph_adapted','Zhang 2024 PU-GN (adapted)')]:
  z=main[(main.variant=='full')&(main.target==t)] if b=='full' else s[(s.baseline==b)&(s.target==t)]
  r[label]='pending' if z.empty else f"{z.iloc[0].fold_mae_mean:.6g} ± {z.iloc[0].fold_mae_sd:.6g}"
 wide.append(r)
pd.DataFrame(wide).to_csv(out/'Table3_fair_baselines.csv',index=False)
md=['# Table 3. Fair external baselines on the reconstructed dataset','', '| '+' | '.join(pd.DataFrame(wide).columns)+' |','|'+'---|'*len(wide[0])]
for r in wide:md.append('| '+' | '.join(r.values())+' |')
md += ['', 'All rows use the same reconstructed dataset, target-specific labels, grouped outer five-fold splits, inner training-only model selection, OOF MAE, and sample SD across folds.', 'Morgan fingerprints are RDKit radius=2, 2048 bits. Morgan-Ridge selects alpha on the inner split; Morgan-RF selects minimum leaf size on the inner split. Mean predictor uses only the outer-training mean.', 'External reported values from other papers are not inserted here because their datasets, targets, units, or metrics differ. See baseline_registry.json and baseline_registry.md.']
md += ['', 'Chen uses the downloaded author getRNNmodel factory with the published best architecture (120 chars, embedding15, two BiLSTM60, TimeDistributedDense30); Keras3 torch backend, fold-local vocabulary and target scaling, inner MSE early stopping (patience10, maximum500 epochs), outer-train refit. The original paper is Tg-only; other targets are explicitly method transfers.',
       'PU-GN uses the downloaded author-team SolubilityGN code/config (3 layers, hidden32, sum aggregation, dropout enabled, Adam .001, L1 .0001). It is an adaptation: pandas and scatter compatibility fixes, per-molecule unit encoding, and one whole-repeat-unit MACCS167 node for failed decompositions. It must not be described as exact reproduction of the original Polymer-Unit Graph paper.',
       'PURS decomposition failed for 2236/8972 structures. Fallback by labeled target: Tg 202/557, FFV 1729/7892, Tc 463/866, Density 340/613, Rg 343/614. Every original test row is retained. Detailed conversion audit and per-fold fallback counts are saved.',
       'Model-selection budgets and stopping criteria differ between methods and are documented; same data and outer folds alone do not establish exhaustive equal-budget optimization. RF currently searches min_samples_leaf=50 or200 only; this restricted RF is preliminary, not evidence of superiority over a well-tuned RF.',
       'Only targets with all five completed outer folds are included as numeric results.']
(out/'Table3_fair_baselines.md').write_text('\n'.join(md)+'\n')
print(pd.DataFrame(wide).to_string(index=False))
