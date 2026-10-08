"""نموذجان (شراء/بيع) يتعلمان من نتائج السوق الحقيقية فقط، مع تحقق walk-forward مُطهَّر (purged)
ومعايرة احتمالات. لا يُفعَّل النموذج إلا إذا تفوّق خارج العينة على العشوائي وحقق توقعاً موجباً."""
import os, joblib, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from . import config
from .ml import FEATS


def make():
    return HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.05, min_samples_leaf=200,
                                          l2_regularization=1.0, max_features=0.6, early_stopping=False,
                                          random_state=7)


def fit(X, y):
    """تدريب على أول 80% ثم معايرة sigmoid على آخر جزء زمني لم يره النموذج."""
    k = int(len(y) * 0.8); gap = config.HORIZON_BARS * 20
    base = make().fit(X[:k], y[:k])
    Xc, yc = X[k + gap:], y[k + gap:]
    if len(yc) < 500 or len(set(yc)) < 2: return base
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    except ImportError:
        cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    return cal.fit(Xc, yc)


def _xy(d, side):
    d = d.dropna(subset=[f"y_{side}"]).sort_values("t")
    return d[FEATS].astype(float).values, d[f"y_{side}"].values.astype(int), d


def walk_forward_auc(d, side, folds=4):
    X, y, _ = _xy(d, side); n = len(y); gap = config.HORIZON_BARS * 20; aucs = []
    for i in range(1, folds + 1):
        tr_end = int(n * (0.4 + 0.15 * (i - 1))); te0, te1 = tr_end + gap, int(n * (0.4 + 0.15 * i))
        if te1 - te0 < 500: continue
        m = make().fit(X[:tr_end], y[:tr_end])
        aucs.append(roc_auc_score(y[te0:te1], m.predict_proba(X[te0:te1])[:, 1]))
    return float(np.mean(aucs)) if aucs else 0.5


def train(d):
    """d: DataFrame من ml.dataset. يحفظ {'long','short','auc_*'}."""
    out, res = {}, {}
    for side in ("long", "short"):
        auc = walk_forward_auc(d, side); res[f"auc_{side}"] = round(auc, 4)
        if auc >= config.MIN_AUC:
            X, y, dd = _xy(d, side); out[side] = fit(X, y)
            thr = pick_threshold(out[side], X, dd[f"r_{side}"].values)
            if thr: out[f"thr_{side}"] = thr; res[f"thr_{side}"] = round(thr, 4)
            else: out.pop(side)
    if out:
        out.update(res); os.makedirs(os.path.dirname(config.MODEL_PATH) or ".", exist_ok=True)
        joblib.dump(out, config.MODEL_PATH)
    res["used"] = sorted(k for k in out if k in ("long", "short"))
    return res


def pick_threshold(m, X, r):
    """عتبة على آخر 20% زمنياً (تقدير): أعلى توقع بعد الرسوم مع 100 صفقة على الأقل؛ None إن لم يوجد توقع موجب."""
    k = int(len(r) * 0.8); p = m.predict_proba(X[k:])[:, 1]; rr = r[k:]; best = (None, 0.0)
    for q in (0.85, 0.9, 0.93, 0.95, 0.97):
        t = float(np.quantile(p, q)); sel = p >= t
        if sel.sum() >= 100:
            ev = float(np.nanmean(rr[sel]) - config.FEE_RT)
            if ev > best[1]: best = (t, ev)
    return best[0]


_cache = {}


def _num(v):
    try: return float(v)
    except (TypeError, ValueError): return np.nan


def load():
    if not os.path.exists(config.MODEL_PATH): return None
    mt = os.path.getmtime(config.MODEL_PATH)
    if _cache.get("mt") != mt:
        try: _cache.update(m=joblib.load(config.MODEL_PATH), mt=mt)
        except Exception: return None
    return _cache["m"]


def predict(feats):
    """يعيد (p_long, p_short) أو None لكل جانب غير مدرب."""
    m = load()
    if not m: return None, None
    x = np.array([[_num(feats.get(k)) for k in FEATS]])  # NaN مسموح: النموذج يتعامل معه أصلاً
    pl = float(m["long"].predict_proba(x)[0, 1]) if "long" in m else None
    ps = float(m["short"].predict_proba(x)[0, 1]) if "short" in m else None
    return pl, ps
