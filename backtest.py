"""باكتست walk-forward صادق (خارج العينة بالكامل) على بيانات Binance الحقيقية.
- يُدرَّب النموذج فقط على الماضي، مع فجوة تطهير، ثم يُختبر على الفترة التالية، ويتدحرج.
- الدخول على افتتاح الشمعة التالية، الخروج Triple-Barrier، رسوم+انزلاق 0.12% لكل صفقة.
- صفقة واحدة مفتوحة لكل عملة في نفس الوقت.
الاستخدام: python backtest.py [BARS]"""
import os, sys, pickle, numpy as np, pandas as pd
from radar import config, ml, model

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "ADAUSDT", "LTCUSDT"]


def load(bars):
    p = f"/tmp/wr_ds_{bars}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    d = ml.dataset(SYMS, bars); pickle.dump(d, open(p, "wb")); return d


def pick_threshold(p, r, min_trades=150):
    """أفضل عتبة احتمال على بيانات التحقق (من الماضي فقط) حسب متوسط العائد بعد الرسوم."""
    best = (None, -1e9)
    for q in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97):
        thr = np.quantile(p, q); m = p >= thr
        if m.sum() < min_trades: continue
        ev = (r[m] - config.FEE_RT).mean()
        if ev > best[1]: best = (thr, ev)
    return best


def simulate(test, side, thr):
    trades = []
    for sym, g in test.groupby("sym"):
        g = g.sort_values("t"); busy = -1
        for i, (p, r, t) in enumerate(zip(g[f"p_{side}"].values, g[f"r_{side}"].values, g.t.values)):
            if i <= busy or not np.isfinite(r) or p < thr: continue
            trades.append((t, sym, side, r - config.FEE_RT)); busy = i + config.HORIZON_BARS
    return trades


def stats(tr, label=""):
    if not tr: return {"set": label, "n": 0}
    tr = sorted(tr); r = np.array([x[3] for x in tr])
    eq = np.cumprod(1 + r * 0.25)              # 25% من رأس المال لكل صفقة
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    sh = r.mean() / (r.std() or 1) * np.sqrt(len(r))
    return {"set": label, "n": len(r), "win%": round((r > 0).mean() * 100, 1), "avg%": round(r.mean() * 100, 3),
            "pf": round(r[r > 0].sum() / abs(r[r < 0].sum() or 1e-9), 2), "equity%": round((eq[-1] - 1) * 100, 1),
            "maxdd%": round(dd * 100, 1), "t_stat": round(sh, 2)}


def main(bars=20000, folds=5):
    d = load(bars).sort_values("t").reset_index(drop=True)
    print("rows", len(d), "symbols", d.sym.nunique(), "from", pd.to_datetime(d.t.min(), unit="ms").date(),
          "to", pd.to_datetime(d.t.max(), unit="ms").date())
    ts = np.sort(d.t.unique()); gap = config.HORIZON_BARS * 15 * 60_000 * 2
    edges = [ts[int(len(ts) * x)] for x in np.linspace(0.45, 1.0, folds + 1)[:-1]] + [ts[-1] + 1]
    all_tr = {"long": [], "short": [], "regime_long": []}
    for k in range(folds):
        t0, t1 = edges[k], edges[k + 1]
        hist = d[d.t < t0 - gap]; test = d[(d.t >= t0) & (d.t < t1)].copy()
        vcut = hist.t.quantile(0.8)
        tr_, va = hist[hist.t < vcut - gap], hist[hist.t >= vcut]
        for side in ("long", "short"):
            X, y, _ = model._xy(tr_, side)
            m = model.make().fit(X, y)
            pv = m.predict_proba(va[model.FEATS].astype(float).values)[:, 1]
            thr, ev = pick_threshold(pv, va[f"r_{side}"].values)
            test[f"p_{side}"] = m.predict_proba(test[model.FEATS].astype(float).values)[:, 1]
            if thr is None or ev <= 0:
                print(f"fold{k} {side}: لا توجد ميزة على التحقق -> لا تداول"); continue
            tr = simulate(test, side, thr); all_tr[side] += tr
            if side == "long":   # فلتر نظام السوق: شراء فقط عندما BTC فوق متوسطه الطويل
                all_tr["regime_long"] += simulate(test[test.btc_trend > 0], side, thr)
            print(f"fold{k} {side}: thr={thr:.3f} val_ev={ev*100:.3f}% ->", stats(tr))
    rows = [stats(all_tr["long"], "شراء فقط"), stats(all_tr["regime_long"], "شراء + فلتر BTC"),
            stats(all_tr["short"], "بيع فقط"), stats(all_tr["long"] + all_tr["short"], "الاثنان")]
    # الأساس: كل الشموع بلا نموذج (عشوائي)
    base = d[d.t >= edges[0]]
    rows.append({"set": "عشوائي شراء (أساس)", "avg%": round((base.r_long - config.FEE_RT).mean() * 100, 3),
                 "win%": round((base.r_long - config.FEE_RT > 0).mean() * 100, 1)})
    out = pd.DataFrame(rows); pd.set_option("display.width", 200)
    print(out.to_string(index=False)); out.to_csv("backtest_results.csv", index=False)
    per = pd.DataFrame(all_tr["regime_long"] or all_tr["long"], columns=["t", "sym", "side", "r"])
    if len(per): print(per.groupby("sym").r.agg(["count", "mean", lambda x: (x > 0).mean()]).round(4))
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
