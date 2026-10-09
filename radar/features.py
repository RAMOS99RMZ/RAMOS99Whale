"""خصائص تدفق الأوامر v3 - كلها مُطبّعة (z-scores / نسب) لتكون قابلة للمقارنة بين العملات.
لا تستخدم أي معلومة من المستقبل: كل نافذة rolling تنظر للخلف فقط."""
import numpy as np, pandas as pd

COLS = "t o h l c v ct qv n tb tbq ig".split()


def kdf(raw):
    df = pd.DataFrame(raw, columns=COLS).astype(float)
    df["sell"] = df.v - df.tb
    df["delta"] = df.tb - df.sell
    return df.reset_index(drop=True)


def atr(df, n=14):
    tr = np.maximum(df.h - df.l, np.maximum(abs(df.h - df.c.shift()), abs(df.l - df.c.shift())))
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def z(s, n=96):
    m, sd = s.rolling(n, min_periods=n // 2).mean(), s.rolling(n, min_periods=n // 2).std()
    return (s - m) / sd.replace(0, np.nan)


def rsi(c, n=14):
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def kline_features(df, ref=None):
    """df: شموع 15m. ref: شموع BTC بنفس الطول (اختياري) لقياس القوة النسبية."""
    f = pd.DataFrame(index=df.index)
    a = atr(df); ap = (a / df.c).replace(0, np.nan)
    vsum = lambda n: df.v.rolling(n).sum().replace(0, np.nan)
    # --- CVD / تدفق الأوامر متعدد الأطر (15m, 1h, 4h, 12h) ---
    for n in (4, 16, 48):
        cvd_n = df.delta.rolling(n).sum() / vsum(n)
        pr_n = df.c.pct_change(n) / (ap * np.sqrt(n))
        f[f"cvd_{n}"] = cvd_n
        f[f"pr_{n}"] = pr_n.clip(-5, 5)
        f[f"div_{n}"] = z(cvd_n, 192) - z(pr_n, 192)          # تباعد CVD عن السعر = تجميع/تصريف خفي
    f["delta_z"] = z(df.delta / df.v.replace(0, np.nan), 96)
    f["cvd_slope"] = f.cvd_16
    f["price_slope"] = f.pr_16
    f["divergence"] = f.div_16
    # --- حجم وحيتان ---
    f["vol_z"] = z(np.log1p(df.qv), 96)
    f["vol_z_4"] = f.vol_z.rolling(4).mean()
    f["trade_size_z"] = z(np.log1p(df.qv / df.n.replace(0, np.nan)), 96)     # متوسط حجم الصفقة
    f["big_buy"] = ((f.trade_size_z > 1) & (df.delta > 0)).astype(float).rolling(16).mean()
    f["big_sell"] = ((f.trade_size_z > 1) & (df.delta < 0)).astype(float).rolling(16).mean()
    rng = (df.h - df.l).replace(0, np.nan)
    wick_lo = (np.minimum(df.o, df.c) - df.l) / rng
    wick_hi = (df.h - np.maximum(df.o, df.c)) / rng
    f["absorption"] = (wick_lo * f.vol_z.clip(lower=0)).rolling(4).mean()      # امتصاص بيع عند القاع
    f["supply"] = (wick_hi * f.vol_z.clip(lower=0)).rolling(4).mean()          # تصريف عند القمة
    # حجم كبير مع حركة صغيرة = امتصاص (effort vs result)
    f["effort_result"] = f.vol_z - z((df.c - df.o).abs() / a.replace(0, np.nan), 96)
    # --- هيكل السعر والاتجاه ---
    for n in (20, 50, 200):
        f[f"ema_{n}"] = (df.c / df.c.ewm(span=n, adjust=False).mean() - 1) / ap
    f["ema_200_slope"] = df.c.ewm(span=200, adjust=False).mean().pct_change(16) / ap
    f["ema_800"] = (df.c / df.c.ewm(span=800, adjust=False).mean() - 1) / ap  # اتجاه ~ 4h EMA200
    f["rsi"] = rsi(df.c) / 100
    f["rsi_1h"] = rsi(df.c, 56) / 100
    hh, ll = df.h.rolling(96).max(), df.l.rolling(96).min()
    f["pos_96"] = (df.c - ll) / (hh - ll).replace(0, np.nan)                  # موقع السعر في نطاق 24h
    f["range_comp"] = (df.h.rolling(16).max() - df.l.rolling(16).min()) / a.replace(0, np.nan) / 16
    f["atr_pct"] = ap
    f["vol_regime"] = z(ap, 384)
    vwap = (df.qv.rolling(96).sum() / vsum(96))
    f["vwap_dist"] = (df.c / vwap - 1) / ap
    f["ret_1"] = df.c.pct_change() / ap
    f["ret_96"] = df.c.pct_change(96) / (ap * 10)
    # --- الوقت ---
    hr = (df.t // 3_600_000) % 24
    f["hr_sin"], f["hr_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    # --- قوة نسبية مقابل BTC ---
    if ref is not None and len(ref) == len(df):
        rap = (atr(ref) / ref.c).replace(0, np.nan)
        f["rs_16"] = f.pr_16 - (ref.c.pct_change(16) / (rap * 4)).clip(-5, 5).values
        f["btc_trend"] = ((ref.c / ref.c.ewm(span=200, adjust=False).mean() - 1) / rap).values
    else:
        f["rs_16"] = 0.0
        f["btc_trend"] = f.ema_200
    return f.replace([np.inf, -np.inf], np.nan)


def book_features(book, price, band=0.01):
    if not book or not book.get("bids") or not book.get("asks"):
        return {"imb_1pct": 0.0, "bid_wall": 0.0, "ask_wall": 0.0}
    b = np.array(book["bids"], float); a = np.array(book["asks"], float)
    bb = b[b[:, 0] >= price * (1 - band)]; aa = a[a[:, 0] <= price * (1 + band)]
    bv, av = (bb[:, 0] * bb[:, 1]).sum(), (aa[:, 0] * aa[:, 1]).sum()
    med = np.median(np.r_[b[:, 0] * b[:, 1], a[:, 0] * a[:, 1]]) or 1
    return {"imb_1pct": float((bv - av) / ((bv + av) or 1)), "bid_wall": float((b[:, 0] * b[:, 1]).max() / med),
            "ask_wall": float((a[:, 0] * a[:, 1]).max() / med)}


def trade_features(trades, whale_usd):
    if not trades: return {"whale_net": 0.0, "whale_cnt": 0, "iceberg": 0.0}
    q = np.array([float(t["q"]) * float(t["p"]) for t in trades]); sell = np.array([bool(t["m"]) for t in trades])
    w = q >= whale_usd; tot = q.sum() or 1
    net = (q[w & ~sell].sum() - q[w & sell].sum()) / tot
    # iceberg: حجم ضخم يُنفَّذ على نفس مستوى السعر مراراً
    s = pd.DataFrame({"p": [t["p"] for t in trades], "q": q}).groupby("p").q.agg(["sum", "count"])
    ice = float((s["sum"].max() / tot) * min(1, s["count"].max() / 20)) if len(s) else 0.0
    return {"whale_net": float(net), "whale_cnt": int(w.sum()), "iceberg": ice}


def spoof_score(hist_rows, side_now_walls):
    """spoofing يحتاج تاريخاً: جدار ظهر في عينتين على الأقل ثم اختفى قبل أن يلمسه السعر."""
    if len(hist_rows) < 3: return 0.0
    seen = {}
    for ts, side, p, q in hist_rows: seen.setdefault((side, round(p, 8)), set()).add(ts)
    vanished = sum(1 for k, v in seen.items() if len(v) >= 2 and k not in side_now_walls)
    return min(1.0, vanished / 5)
