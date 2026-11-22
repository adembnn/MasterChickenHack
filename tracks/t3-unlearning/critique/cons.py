import pandas as pd, numpy as np
d=pd.read_csv("sel.csv"); N=len(d)
r=lambda v: v.rank(pct=True)
d["consensus"]=(r(d.pred_depth)+r(d.oof_b8_loss)+r(-d.margin))/3
def vote(row):
    c=[row.orig_second]
    if row.knn_wrong_class>=0: c.append(row.knn_wrong_class)
    if row.oof_b8_pred!=row.orig_pred: c.append(row.oof_b8_pred)
    v=pd.Series(c).value_counts(); return int(v.index[0]), int(v.iloc[0])
vv=d.apply(vote,axis=1); d["vote_class"]=[a for a,b in vv]; d["vote_n"]=[b for a,b in vv]
d.to_csv("sel.csv",index=False,float_format="%.4f")
top=d.nlargest(int(.3*N),"consensus")
print("top30 consensus: vote_n>=2", (top.vote_n>=2).mean().round(3), "vote_class==orig_second", (top.vote_class==top.orig_second).mean().round(3))
print("depth>=10 share in top30", (top.pred_depth>=10).mean().round(3), "oof wrong share", (top.oof_b8_pred!=top.orig_pred).mean().round(3))
print("classes in top30", top.orig_pred.nunique(), "max per class", top.orig_pred.value_counts().max())
