"""Baseline comparison for Few-Shot Bayesian Calibration (same LOLO protocol, same random supports).
Needs bayes_fewshot.py in the same folder.   usage: python baseline_comparison.py [csv]
"""
import sys, warnings, numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
warnings.filterwarnings("ignore")
import bayes_fewshot as bf

CSV = sys.argv[1] if len(sys.argv) > 1 else "4_Final_Dataset/final_regression_dataset.csv"

# ---------------------------------------------------------------- cup table (+ extra features)
def build(path):
    d = pd.read_csv(path); c = bf.build_cups(path)
    img = ["Whole_Grains_Count", "Uniformity_Rate_Pct", "Grain_Length_mm_Mean", "Grain_Width_mm_Mean",
           "Grain_Thickness_mm_Mean", "Grain_Area_mm2_Mean", "Grain_Volume_mm3_Mean", "Grain_Length_mm_Std",
           "Grain_Width_mm_Std"]
    ex = d.groupby("Physical_Sample_ID").agg(**{k: (k, "mean") for k in img},
        W=("Weight_g", "first"), Hr=("Rice_Height_mm", "first"), phys=("Physical_Estimated_Seeds", "mean")).reset_index()
    c = c.merge(ex, left_on="cup", right_on="Physical_Sample_ID").drop(columns="Physical_Sample_ID")
    c["lW"] = np.log(c.W); c["lH"] = np.log(c.Hr); c["lphys"] = np.log(c.phys)
    c["lcnt"] = np.log(c.Whole_Grains_Count.clip(lower=1))
    bad = c.W / c.N > 0.1
    print("excluded (invalid weight): %d cups -> %d cups, lots %s" % (bad.sum(), (~bad).sum(), c[~bad].lot.value_counts().sort_index().to_dict()))
    return c[~bad].reset_index(drop=True), img

c, IMG = build(CSV)
BULK = ["lV", "lD", "lH"]; ALL = BULK + IMG
SW = 10.0                                       # sample weight of each support cup in "pooled + support" baselines

# ---------------------------------------------------------------- baselines
def lin(src, cols, sup=None, w=None):
    d = src if sup is None else pd.concat([src, sup]); sw = None if sup is None else np.r_[np.ones(len(src)), np.full(len(sup), SW)]
    return LinearRegression().fit(d[cols], d.y, sample_weight=sw)
def shift(m, cols, sup): return (sup.y - m.predict(sup[cols])).mean()

def P1(src, sup, q, K): return q.phys.values
def P2(src, sup, q, K): return np.exp(q.lphys + (sup.y - sup.lphys).mean()).values
def V1(src, sup, q, K): return np.exp(make_pipeline(StandardScaler(), Ridge(1.0)).fit(src[IMG], src.y).predict(q[IMG]))
def W1(src, sup, q, K): return np.exp(q.lW + (src.y - src.lW).mean()).values
def W2(src, sup, q, K): return np.exp(q.lW + (sup.y - sup.lW).mean()).values
def W3(src, sup, q, K): m = lin(src, ["lW", "lV"]); return np.exp(m.predict(q[["lW", "lV"]]))
def W4(src, sup, q, K): m = lin(src, ["lW", "lV"]); return np.exp(m.predict(q[["lW", "lV"]]) + shift(m, ["lW", "lV"], sup))
def T1(src, sup, q, K):
    d = pd.concat([src, sup]); sw = np.r_[np.ones(len(src)), np.full(len(sup), SW)]
    m = make_pipeline(StandardScaler(), Ridge(1.0)); m.fit(d[ALL], d.y, ridge__sample_weight=sw); return np.exp(m.predict(q[ALL]))
def _tree(mk, src, sup, q):
    d = pd.concat([src, sup]); sw = np.r_[np.ones(len(src)), np.full(len(sup), SW)]
    return np.exp(mk().fit(d[ALL], d.y, sample_weight=sw).predict(q[ALL]))
def T2(src, sup, q, K): return _tree(lambda: RandomForestRegressor(100, min_samples_leaf=2, random_state=0, n_jobs=-1), src, sup, q)
def T3(src, sup, q, K): return _tree(lambda: GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=0), src, sup, q)
def _gp(): return make_pipeline(StandardScaler(), GPR(C(1.0) * RBF(2.0) + WhiteKernel(1e-2, (1e-6, 1)), normalize_y=True, random_state=0))
def G1(src, sup, q, K): return np.exp(_gp().fit(sup[["lV"]], sup.y).predict(q[["lV"]]))
def G2(src, sup, q, K):
    d = pd.concat([src, sup]); sw = np.r_[np.ones(len(src)), np.full(len(sup), SW)]
    m = lin(src, ["lV"]); r = d.y - m.predict(d[["lV"]])                     # GP on residual of the source log-log line
    g = _gp(); g.fit(d[["lV"]], r)
    return np.exp(m.predict(q[["lV"]]) + g.predict(q[["lV"]]))

BASELINES = [  # (group, name, fn, min_K)   K-free methods are marked
 ("Physics-based",   "Repo physical estimate (K-free)",             P1, 0),
 ("Physics-based",   "Repo physical estimate + MLE scale",          P2, 1),
 ("Physics-based",   "Physics (V_bulk/V_g), source offset (K-free)", bf.b_physics_zero_w if hasattr(bf, "b_physics_zero_w") else lambda s, u, q, K: bf.b_physics_zero(s, u, q), 0),
 ("Vision-only",     "Ridge, image features only (K-free)",         V1, 0),
 ("Weight-based",    "Weight only, source scale (K-free)",          W1, 0),
 ("Weight-based",    "Weight only + MLE scale",                     W2, 1),
 ("Weight-based",    "Weight + V_bulk log-log (K-free)",            W3, 0),
 ("Weight-based",    "Weight + V_bulk log-log + MLE shift",         W4, 1),
 ("Volume log-log",  "V_bulk log-log, pooled source (K-free)",      lambda s, u, q, K: bf.b_logv_zero(s, u, q), 0),
 ("Volume log-log",  "V_bulk log-log, source slope + MLE shift",    lambda s, u, q, K: bf.b_logv_shift(s, u, q), 1),
 ("Volume log-log",  "V_bulk log-log, support-only (from scratch)", lambda s, u, q, K: bf.b_support_only(s, u, q), 3),
 ("Tabular ML",      "Ridge, all features, pooled + support",       T1, 0),
 ("Tabular ML",      "Random Forest, all features, pooled + support", T2, 0),
 ("Tabular ML",      "Gradient Boosting, all features, pooled + support", T3, 0),
 ("Gaussian process","GP on V_bulk, support-only",                  G1, 3),
 ("Gaussian process","GP residual on source log-log, pooled + support", G2, 0),
]
OURS = {"Ours: Bayesian calibration, g(z)=V_g": dict(z_cols=("o",)),
        "Ours: Bayesian calibration, no V_g term": dict(z_cols=("lD", "lasp"), use_vg=False),
        "Ours+W: g(z)=V_g, weight":              dict(z_cols=("o", "lW")),
        "Ours+W: g(z)=V_g, weight, V_bulk":      dict(z_cols=("o", "lW", "lV")),
        "Ours+W: no V_g term, weight, V_bulk":   dict(z_cols=("lW", "lV"), use_vg=False),
        "Ours-W: physics term = log W (scale only)": dict(z_cols=(), fit_g=False, use_vg=False, phys_col="lW"),
        "Ours-W: physics term = log W, g(z)=V_g":    dict(z_cols=("o",), use_vg=False, phys_col="lW"),
        "Ours-W: physics term = log W, g(z)=V_g,V_bulk": dict(z_cols=("o", "lV"), use_vg=False, phys_col="lW")}

def run(Ks=(0, 1, 3, 5, 10, 20), R=50, seed=0, min_query=5):
    rng = np.random.default_rng(seed); rows = []
    for lot in sorted(c.lot.unique()):
        tgt = c[c.lot == lot].reset_index(drop=True); src = c[c.lot != lot].reset_index(drop=True)
        fit = {k: bf.BayesLotCalibrator(**v).fit_source(src) for k, v in OURS.items()}
        for K in Ks:
            if len(tgt) - K < min_query: continue
            for r in range(1 if K == 0 else R):
                idx = rng.permutation(len(tgt)); sup, q = tgt.iloc[idx[:K]], tgt.iloc[idx[K:]]
                for g, n, fn, mk in BASELINES:
                    if K >= mk: rows.append(dict(group=g, model=n, lot=lot, K=K, rep=r, **bf.metrics(q.N.values, np.asarray(fn(src, sup, q, K)))))
                for n, m in fit.items():
                    p, lo, hi = m.adapt(sup).predict(q); rows.append(dict(group="Ours", model=n, lot=lot, K=K, rep=r, **bf.metrics(q.N.values, p, lo, hi)))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    res = run(); res.to_csv("baseline_raw.csv", index=False)
    cols = ["MAE", "MSE", "RMSE", "MAPE", "R2", "Cov90"]
    macro = res.groupby(["group", "model", "K", "lot"])[cols].mean().groupby(["group", "model", "K"]).mean().reset_index()
    macro.round(3).to_csv("baseline_macro.csv", index=False)
    order = [g for g, *_ in BASELINES if True]; order = list(dict.fromkeys(order)) + ["Ours"]  # Ours incl. Ours+W
    # ---- compact paper table: MAE / RMSE / MAPE at K = 0, 1, 5, 10
    Ks = [0, 1, 5, 10]; lines = []
    hdr = "| Group | Method | " + " | ".join("K=%d MAE / RMSE / MAPE%%" % k for k in Ks) + " |"
    lines += [hdr, "|" + "---|" * (2 + len(Ks))]
    for g in order:
        for m in macro[macro.group == g].model.unique():
            cells = []
            for k in Ks:
                r = macro[(macro.model == m) & (macro.K == k)]
                cells.append("–" if r.empty else "%.1f / %.1f / %.2f" % (r.MAE.iloc[0], r.RMSE.iloc[0], r.MAPE.iloc[0]))
            lines.append("| %s | %s | %s |" % (g, m, " | ".join(cells)))
    open("paper_table.md", "w").write("\n".join(lines)); print("\n".join(lines))
    # ---- paired comparison vs Ours (K=5, 10): fraction of draws in which Ours has lower MAPE (macro over lots per draw)
    ref = "Ours: Bayesian calibration, g(z)=V_g"; print("\nPaired vs", ref)
    for K in [3, 5, 10]:
        p = res[res.K == K].groupby(["model", "rep", "lot"]).MAPE.mean().groupby(["model", "rep"]).mean().unstack(0)
        out = []
        for m in p.columns:
            if m == ref: continue
            diff = (p[m] - p[ref]).dropna()
            out.append((m, diff.mean(), 100 * (diff > 0).mean(), np.percentile(diff, 2.5), np.percentile(diff, 97.5)))
        print("K=%d  (positive = Ours better)" % K)
        for m, dm, w, lo, hi in sorted(out, key=lambda x: x[1]): print("   %-58s dMAPE=%+6.2f [%+.2f,%+.2f]  Ours wins %3.0f%%" % (m, dm, lo, hi, w))
