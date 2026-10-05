from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parent;out=root/'results';s=pd.read_csv(out/'classical_performance.csv');main=pd.read_csv(root.parent/'training/results/performance_summary.csv');
wide=[]
for t in ['Tg','FFV','Tc','Density','Rg']:
 r={'Property':t}
 for b,label in [('full','Poly-Prophet recomputed'),('mean','Mean predictor'),('morgan_ridge','Morgan-Ridge'),('morgan_rf','Morgan-RF')]:
  z=main[(main.variant=='full')&(main.target==t)] if b=='full' else s[(s.baseline==b)&(s.target==t)]
  r[label]='pending' if z.empty else f"{z.iloc[0].fold_mae_mean:.6g} ± {z.iloc[0].fold_mae_sd:.6g}"
 wide.append(r)
pd.DataFrame(wide).to_csv(out/'Table3_fair_baselines.csv',index=False)
md=['# Table 3. Fair external baselines on the reconstructed dataset','', '| '+' | '.join(pd.DataFrame(wide).columns)+' |','|'+'---|'*len(wide[0])]
for r in wide:md.append('| '+' | '.join(r.values())+' |')
md += ['', 'All rows use the same reconstructed dataset, target-specific labels, grouped outer five-fold splits, inner training-only model selection, OOF MAE, and sample SD across folds.', 'Morgan fingerprints are RDKit radius=2, 2048 bits. Morgan-Ridge selects alpha on the inner split; Morgan-RF selects minimum leaf size on the inner split. Mean predictor uses only the outer-training mean.', 'External reported values from other papers are not inserted here because their datasets, targets, units, or metrics differ. See baseline_registry.json and baseline_registry.md.']
(out/'Table3_fair_baselines.md').write_text('\n'.join(md)+'\n')
print(pd.DataFrame(wide).to_string(index=False))
