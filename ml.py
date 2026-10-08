"""بناء مجموعة التدريب من التاريخ: وسم Triple-Barrier لكل شمعة (لا تعلّم من التنبؤات الذاتية).
هذا يسمح بتدريب النموذج من اليوم الأول على آلاف الأمثلة الحقيقية بدل انتظار أسابيع."""
import numpy as np, pandas as pd
from . import config, features as F, sources

FEATS = ["cvd_4", "cvd_16", "cvd_48", "pr_4", "pr_16", "pr_48", "div_4", "div_16", "div_48", "delta_z",
         "vol_z", "vol_z_4", "trade_size_z", "big_buy", "big_sell", "absorption", "supply", "effort_result",
         "ema_20", "ema_50", "ema_200", "ema_200_slope", "ema_800", "rsi", "rsi_1h", "pos_96", "range_comp",
         "atr_pct", "vol_regime", "vwap_dist", "ret_1", "ret_96", "hr_sin", "hr_cos", "rs_16", "btc_trend"]


def history(sym, n=20000, interval="15m"):
    out, end = [], None
    while len(out) < n:
        k = sources.klines(sym, interval, 1000, end)
        if not k: break
        out = k + out; end = k[0][0] - 1
        if len(k) < 1000: break
    return out[-n:]


def triple_barrier(df, side, tp=None, sl=None, horizon=None):
    """لكل شمعة: 1 إذا لُمس الهدف قبل الوقف خلال المهلة. الدخول على افتتاح الشمعة التالية (واقعي)."""
    tp, sl, horizon = tp or config.TP_ATR, sl or config.SL_ATR, horizon or config.HORIZON_BARS
    a = F.atr(df).values; o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    n = len(df); y = np.full(n, np.nan); r = np.full(n, np.nan)
    for i in range(n - horizon - 1):
        e = o[i + 1]; d = a[i]
        if not np.isfinite(d) or d <= 0: continue
        up, dn = (e + tp * d, e - sl * d) if side == 1 else (e + sl * d, e - tp * d)
        res = None
        for j in range(i + 1, i + 1 + horizon):
            if side == 1:
                if l[j] <= dn: res = (0, -sl * d / e); break
                if h[j] >= up: res = (1, tp * d / e); break
            else:
                if h[j] >= up: res = (0, -sl * d / e); break
                if l[j] <= dn: res = (1, tp * d / e); break
        if res is None:
            ex = c[i + horizon]; rr = (ex - e) / e * side; res = (int(rr > 0), rr)
        y[i], r[i] = res
    return y, r


def build(sym, raw, ref_raw=None):
    df = F.kdf(raw)
    ref = None
    if ref_raw is not None:
        rdf = F.kdf(ref_raw).set_index("t").reindex(df.t).ffill().reset_index()
        if rdf.c.notna().all(): ref = rdf
    f = F.kline_features(df, ref)
    yl, rl = triple_barrier(df, 1); ys, rs = triple_barrier(df, -1)
    f["y_long"], f["r_long"], f["y_short"], f["r_short"] = yl, rl, ys, rs
    f["t"], f["sym"], f["close"] = df.t.values, sym, df.c.values
    return f.iloc[config.WARMUP:].reset_index(drop=True)


def dataset(symbols, n=20000):
    ref = history("BTCUSDT", n)
    frames = []
    for s in symbols:
        raw = ref if s == "BTCUSDT" else history(s, n)
        if len(raw) < config.WARMUP + 500: continue
        frames.append(build(s, raw, ref))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
