"""Exact atlas-level sign-flip tests for the added-gene comparison, with Holm-adjusted P values (Fig. 6d, Supplementary
Data).  Same per-task results and the same cluster bootstrap as panel_tables/added_genes_tests.py, plus the exact
sign-flip test over the 11 parent atlases; nothing is re-scored.
Out: paper/added_gene_tests/data/exact_tests.csv"""
import glob, itertools, re
import numpy as np, pandas as pd
SB="/path/to/research/analysis"; R=f"{SB}/program_scoring/added_genes/results"; B=10_000
E=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/eval_B/*.csv"))],ignore_index=True)
E=E[(E.scheme=="donor")&(E.calibration=="matched")].copy()
E["dose"]=E.variant.map(lambda v:0 if v=="orig" else int(re.match(r"\w+_k(\d+)_d\d$",v).group(1)))
parent=pd.read_csv(f"{SB}/paper/program_scoring_tables/data/reuse_by_task.csv").set_index("dataset").parent
tasks=sorted(E.task.unique()); par=sorted({parent[t] for t in tasks}); pidx=np.array([par.index(parent[t]) for t in tasks]); ntask=np.bincount(pidx,minlength=len(par))
draws=np.random.default_rng(0).integers(0,len(par),size=(B,len(par))); C=np.stack([(draws==j).sum(1) for j in range(len(par))],1); denom=C@ntask
SIGNS=np.array(list(itertools.product([1,-1],repeat=len(par))))
def infer(d):
    m=(C@np.bincount(pidx,weights=d,minlength=len(par)))/denom; lo,hi=np.percentile(m,[2.5,97.5])
    pb=max(2*min((m<=0).mean(),(m>=0).mean()),2/B); S=np.bincount(pidx,weights=d,minlength=len(par)); pf=(np.abs(SIGNS@S)>=abs(S.sum())-1e-12).mean()
    return dict(mean=d.mean()*100,lo=lo*100,hi=hi*100,p_boot=pb,p_flip=pf,tasks_pos=int((d>0).sum()),tasks_neg=int((d<0).sum()))
def holm(ps):
    ps=np.asarray(ps,float);o=np.argsort(ps);m=len(ps);a=np.empty(m);run=0
    for r,i in enumerate(o):
        run=max(run,(m-r)*ps[i]);a[i]=min(1,run)
    return a
rows=[]
for met in ("balanced_acc","macro_f1","mean_auroc"):
    T=E.groupby(["dose","task","method"])[met].mean()
    base=T.loc[0].unstack("method").loc[tasks]
    for k in (0,5,10,20,40):
        W=T.loc[k].unstack("method").loc[tasks]
        for m in W.columns:
            if m=="RECAST": continue
            rows.append(dict(metric=met,k=k,test="absolute",comparator=m,**infer((W.RECAST-W[m]).values)))
            if k: rows.append(dict(metric=met,k=k,test="change",comparator=m,**infer(((W.RECAST-base.RECAST)-(W[m]-base[m])).values)))
O=pd.DataFrame(rows); O.to_csv(f"{SB}/paper/added_gene_tests/data/exact_tests.csv",index=False)
pd.set_option('display.width',200);pd.set_option('display.max_rows',300)
print("RECAST above all 18 comparators, absolute score (RECAST minus comparator, points): smallest bootstrap CI lower bound, largest exact P")
a=O[O.test=="absolute"]
for (met,k),g in a.groupby(["metric","k"]):
    w=g.loc[g.p_flip.idxmax()]; w2=g.loc[g.lo.idxmin()]
    print(f"{met:13s} k={k:2d}  CI above zero {int((g.lo>0).sum()):2d}/18  exact P<0.05 {int((g.p_flip<.05).sum()):2d}/18  largest exact P {w.p_flip:.4f} ({w.comparator})  lowest CI bound {w2.lo:+.2f} ({w2.comparator})")
print("\nChange (RECAST minus comparator): exact P<0.05 out of 18, and which fail")
c=O[O.test=="change"]
for (met,k),g in c.groupby(["metric","k"]):
    print(f"{met:13s} k={k:2d} boot {int((g.p_boot<.05).sum()):2d}/18  exact {int((g.p_flip<.05).sum()):2d}/18  fail: {[(r.comparator,round(r.p_flip,3)) for r in g[g.p_flip>=.05].itertuples()]}")
print("\nThe eight displayed in Fig. 6d, per metric")
for met in ("balanced_acc","macro_f1","mean_auroc"):
    x=c[(c.metric==met)&c.comparator.isin(["SIGnature","ANS"])].sort_values(["comparator","k"]).copy()
    x["holm_boot"]=holm(x.p_boot); x["holm_flip"]=holm(x.p_flip)
    print(met);print(x[["comparator","k","mean","p_boot","holm_boot","p_flip","holm_flip","tasks_pos","tasks_neg"]].round(4).to_string(index=False))
print("\nAbsolute, SIGnature and ANS, BA"); y=a[(a.metric=="balanced_acc")&a.comparator.isin(["SIGnature","ANS"])].sort_values(["comparator","k"])
print(y[["comparator","k","mean","lo","hi","p_boot","p_flip"]].round(4).to_string(index=False))
