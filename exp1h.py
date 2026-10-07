import sys
sys.path.insert(0, '.')
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.features_v2 import add_features_v2
from src.indicators import add_indicators
from src.ingest_ext import fetch_ohlcv_paginated

FEATS = ['trend_strength','price_vs_ema12','rsi14','stoch_k','macd_hist','roc_5','roc_10','roc_20','bb_pct','bb_width','atr_pct','vol_20','adx14','vol_ratio','obv_slope','high_low_range','close_pos_range','emp_p_up_50','rsi_percentile']
def wilson(p,n,z=1.96):
    if n==0: return 0
    d=1+z*z/n; c=p+z*z/(2*n); m=z*np.sqrt((p*(1-p)+z*z/(4*n))/n)
    return (c-m)/d
raw = fetch_ohlcv_paginated('BTC/USDT','1h',total=5000)
raw['is_closed']=True
df = add_features_v2(add_indicators(raw))
for H in [3, 6, 12, 24]:
    d = df.copy()
    d['target'] = (d['close'].shift(-H) > d['close']).astype(int)
    d = d.dropna(subset=FEATS+['target']).reset_index(drop=True)
    X, y = d[FEATS], d['target'].astype(int)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,shuffle=False,random_state=7)
    base = RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=15,class_weight='balanced',random_state=7,n_jobs=-1)
    cal = CalibratedClassifierCV(base,method='isotonic',cv=3)
    cal.fit(Xtr,ytr)
    p = cal.predict_proba(Xte)[:,1]
    best=(0,0,0,0)
    for thr in [0.55,0.6,0.65,0.7,0.75,0.8]:
        m = p>=thr
        if m.sum()>=15:
            prec=(yte[m]==1).mean(); lb=wilson(prec,int(m.sum()))
            if prec>best[0]: best=(prec,int(m.sum()),thr,lb)
    print(f'1h H={H} n_test={len(yte)} base={y.mean():.2f} best prec={best[0]:.1%} n={best[1]} thr={best[2]} LB={best[3]:.1%}')
