"""Few-Shot Bayesian Calibration with Uncertain Visual Geometry  (bulk rice grain counting)

    log N = log V_bulk - log V_g + b_lot + g(z)

 * V_bulk : cylinder volume from measured cup diameter + fill height (known, not learned)
 * V_g    : grain volume from vision, UNCERTAIN (few whole grains/view -> noisy)
 * g(z)   : shared residual model (ridge), trained on source lots, FIXED at adaptation
 * b_lot  : task parameter, Gaussian prior from source lots, conjugate posterior from K support cups
 * quality-aware noise: var_i = s0^2 + sg^2 / n_i   (n_i = whole grains measured in cup i)

Protocol: Leave-One-Lot-Out. Lot = cup type (Inner_Diameter, no variety/lot label exists in the data).
Support and query are DIFFERENT physical cups of the target lot. Source lots never see target labels.

usage: python bayes_fewshot.py [path/to/final_regression_dataset.csv]
"""
import sys, warnings, numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import norm
from sklearn.linear_model import Ridge, LinearRegression
warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "4_Final_Dataset/final_regression_dataset.csv"
Z90 = norm.ppf(0.95)

# ------------------------------------------------------------------ data -> cup table
def build_cups(path):
    d = pd.read_csv(path)
    d["lvg"] = np.log(d["Mean_Clean_Grain_Volume_mm3"])
    d["n"] = d["Whole_Grains_Filtered_Count"].clip(lower=1)
    rows = []
    for pid, s in d.groupby("Physical_Sample_ID"):
        n = s["n"].sum()
        rows.append(dict(
            cup=pid, N=float(s.Actual_Count.iloc[0]),
            lV=np.log(s.Bulk_Rice_Volume_mm3.iloc[0]),                    # log V_bulk (known)
            o=float(np.average(s.lvg, weights=s.n)),                      # visual log V_g (uncertain)
            n=float(n), views=len(s),
            lD=np.log(s.Inner_Diameter_mm.iloc[0]),
            lasp=np.log(s.Rice_Height_mm.iloc[0] / s.Inner_Diameter_mm.iloc[0]),
            lot=int(round(s.Inner_Diameter_mm.iloc[0]))))                 # 17.8,22.8,{32.6,32.8},41.5 -> 4 lots
    c = pd.DataFrame(rows)
    c["y"] = np.log(c.N)
    return c

# ------------------------------------------------------------------ the framework
class BayesLotCalibrator:
    """z_cols: features of g(z); use_vg: include -log V_g (physics term); quality: heteroscedastic noise;
       tau_scale: inflate prior sd of b_lot (sensitivity)."""
    def __init__(self, z_cols=("o",), use_vg=True, quality=True, fit_g=True, alpha=1.0, tau_scale=1.0, phys_col="lV"):
        self.phys_col = phys_col
        self.z, self.use_vg, self.quality, self.fit_g, self.alpha, self.tau_scale = list(z_cols), use_vg, quality, fit_g, alpha, tau_scale

    def _phys(self, c):                        # physics part: log V_bulk - log V_g
        return c[self.phys_col].values - (c.o.values if self.use_vg else 0.0)

    def fit_source(self, src):                 # ---- shared model + prior, trained on source lots only
        r = src.y.values - self._phys(src)     # what b_lot + g(z) must explain
        lots = src.lot.values
        if self.fit_g and self.z:
            Z = src[self.z].values
            self.mu_z, self.sd_z = Z.mean(0), Z.std(0) + 1e-9
            Zs = (Z - self.mu_z) / self.sd_z
            Zw = Zs.copy(); rw = r.copy()      # within-lot (fixed-effect) estimator: g must not absorb lot offsets
            for l in np.unique(lots):
                m = lots == l; Zw[m] -= Zs[m].mean(0); rw[m] -= r[m].mean()
            self.g = Ridge(alpha=self.alpha, fit_intercept=False).fit(Zw, rw)
            gz = self.g.predict(Zs)
        else:
            self.g = None; gz = np.zeros(len(src))
        res = r - gz
        bl = np.array([res[lots == l].mean() for l in np.unique(lots)])          # per-source-lot offsets
        self.b0 = bl.mean()
        self.tau = max(bl.std(ddof=1), 0.05) * self.tau_scale                    # prior sd of b_lot
        e2 = np.concatenate([(res[lots == l] - res[lots == l].mean()) ** 2 for l in np.unique(lots)])
        ordr = np.concatenate([np.where(lots == l)[0] for l in np.unique(lots)])
        if self.quality:                                                         # var = s0^2 + sg^2 / n  (NNLS)
            A = np.c_[np.ones(len(ordr)), 1.0 / src.n.values[ordr]]
            (s0, sg), _ = nnls(A, e2); self.s0, self.sg = max(s0, 1e-4), sg
        else:
            self.s0, self.sg = e2.mean(), 0.0
        return self

    def _gz(self, c):
        if self.g is None: return np.zeros(len(c))
        return self.g.predict((c[self.z].values - self.mu_z) / self.sd_z)

    def _var(self, c): return self.s0 + self.sg / c.n.values

    def adapt(self, sup):                      # ---- conjugate Normal update of b_lot from K labelled cups
        if len(sup) == 0:
            self.mu_b, self.var_b = self.b0, self.tau ** 2; return self
        rs = sup.y.values - self._phys(sup) - self._gz(sup); w = 1.0 / self._var(sup)
        P = 1.0 / self.tau ** 2 + w.sum()
        self.mu_b = (self.b0 / self.tau ** 2 + (w * rs).sum()) / P; self.var_b = 1.0 / P
        return self

    def predict(self, q):                      # ---- p(N_q | x_q, S_K): lognormal, input uncertainty propagated
        m = self._phys(q) + self._gz(q) + self.mu_b
        v = self._var(q) + self.var_b
        sd = np.sqrt(v)
        return np.exp(m), np.exp(m - Z90 * sd), np.exp(m + Z90 * sd)

# ------------------------------------ baselines (same support/query)
def b_physics_zero(src, sup, q):               # source-mean offset, raw V_g, no adaptation
    off = (src.y - src.lV + src.o).mean(); return np.exp(q.lV - q.o + off).values
def b_physics_mle(src, sup, q):                # physics + mean residual of support (no prior, no g)
    return np.exp(q.lV - q.o + (sup.y - sup.lV + sup.o).mean()).values
def b_logv_zero(src, sup, q):                  # pooled source log-log ridge on log V_bulk
    m = LinearRegression().fit(src[["lV"]], src.y); return np.exp(m.predict(q[["lV"]]))
def b_logv_shift(src, sup, q):                 # source slope + MLE intercept shift from support
    m = LinearRegression().fit(src[["lV"]], src.y)
    return np.exp(m.predict(q[["lV"]]) + (sup.y - m.predict(sup[["lV"]])).mean())
def b_support_only(src, sup, q):               # train from scratch on K support cups
    m = LinearRegression().fit(sup[["lV"]], sup.y); return np.exp(m.predict(q[["lV"]]))

BASE = {  # name: (fn, min_K)
    "B1 Physics, source offset (K-free)": (b_physics_zero, 0),
    "B2 Physics + MLE offset":            (b_physics_mle, 1),
    "B3 log-log pooled source (K-free)":  (b_logv_zero, 0),
    "B4 log-log source slope + shift":    (b_logv_shift, 1),
    "B5 log-log support-only":            (b_support_only, 3),
}
OURS = {
    "Ours (g=0)":                  dict(z_cols=(), fit_g=False),
    "Ours (g(z): V_g)":            dict(z_cols=("o",)),
    "Ours (g(z): V_g+cup)":        dict(z_cols=("o", "lD", "lasp")),
    "Ours (g(z): V_g+cup) const-var": dict(z_cols=("o", "lD", "lasp"), quality=False),
    "Ours (no V_g term)":          dict(z_cols=("lD", "lasp"), use_vg=False),
}

def metrics(N, p, lo=None, hi=None):
    e = N - p
    out = dict(MAE=np.mean(np.abs(e)), MSE=np.mean(e ** 2), RMSE=np.sqrt(np.mean(e ** 2)),
               MAPE=100 * np.mean(np.abs(e) / N), R2=1 - np.sum(e ** 2) / np.sum((N - N.mean()) ** 2))
    if lo is not None:
        out["Cov90"] = 100 * np.mean((N >= lo) & (N <= hi)); out["Width%"] = 100 * np.mean((hi - lo) / N)
    return out

def run(c, Ks=(0, 1, 3, 5, 10, 20), R=200, seed=0, min_query=5):
    rng = np.random.default_rng(seed); rows = []
    for lot in sorted(c.lot.unique()):
        tgt = c[c.lot == lot].reset_index(drop=True); src = c[c.lot != lot].reset_index(drop=True)
        fitted = {k: BayesLotCalibrator(**v).fit_source(src) for k, v in OURS.items()}
        for K in Ks:
            if len(tgt) - K < min_query: continue
            for r in range(1 if K == 0 else R):
                idx = rng.permutation(len(tgt)); sup, q = tgt.iloc[idx[:K]], tgt.iloc[idx[K:]]
                for name, (fn, mk) in BASE.items():
                    if K >= mk: rows.append(dict(lot=lot, K=K, rep=r, model=name, **metrics(q.N.values, fn(src, sup, q))))
                for name, m in fitted.items():
                    p, lo, hi = m.adapt(sup).predict(q)
                    rows.append(dict(lot=lot, K=K, rep=r, model=name, **metrics(q.N.values, p, lo, hi)))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    c = build_cups(CSV)
    print("cups: %d | lots(cup diam mm): %s" % (len(c), c.lot.value_counts().sort_index().to_dict()))
    res = run(c); res.to_csv("bayes_results_raw.csv", index=False)
    cols = ["MAE", "MSE", "RMSE", "MAPE", "R2", "Cov90", "Width%"]
    per_lot = res.groupby(["model", "K", "lot"])[cols].mean().reset_index()
    macro = per_lot.groupby(["model", "K"])[cols].mean().round(2).reset_index()      # macro-avg over target lots
    macro.to_csv("bayes_results_macro.csv", index=False); per_lot.round(3).to_csv("bayes_results_per_lot.csv", index=False)
    pd.set_option("display.width", 200)
    for met in ["MAPE", "MAE", "RMSE"]:
        print("\n%s (macro-avg over 4 held-out lots)" % met); print(macro.pivot(index="model", columns="K", values=met).to_string())
    print("\nCoverage of nominal-90%% interval, and mean relative width"); 
    print(macro.dropna(subset=["Cov90"]).pivot(index="model", columns="K", values="Cov90").to_string())
    print(macro.dropna(subset=["Width%"]).pivot(index="model", columns="K", values="Width%").to_string())
    print("\nPer-lot MAPE at K=5"); print(per_lot[per_lot.K == 5].pivot(index="model", columns="lot", values="MAPE").round(2).to_string())
