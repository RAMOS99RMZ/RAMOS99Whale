"""حلقة التشغيل: وسم الإشارات القديمة ← (إعادة تدريب من التاريخ كل 6 ساعات) ← 8 عينات لحظية بفاصل دقيقة."""
import time, json, sys, os
from . import config, db, sources, features as F, detectors, model, labeler, notifier, ml


def scan_symbol(c, sym, now, ref_raw):
    raw = sources.klines(sym, "15m", 1000)
    if len(raw) < config.WARMUP: return None
    df = F.kdf(raw)
    ref = None
    if ref_raw:
        r = F.kdf(ref_raw).set_index("t").reindex(df.t).ffill().reset_index()
        if r.c.notna().all(): ref = r
    kf = F.kline_features(df, ref).iloc[-1].to_dict()
    price = float(df.c.iloc[-1]); atr = float(F.atr(df).iloc[-1])
    book = sources.depth(sym); bf = F.book_features(book, price)
    whale_usd = max(50_000, float(df.qv.tail(96).mean()) / 200)
    tf = F.trade_features(sources.agg_trades(sym, (now - config.SAMPLE_SECONDS) * 1000), whale_usd)
    sp = 0.0
    if book:
        walls = {("B", round(float(p), 8)) for p, q in book["bids"] if float(p) * float(q) > whale_usd} | \
                {("A", round(float(p), 8)) for p, q in book["asks"] if float(p) * float(q) > whale_usd}
        c.executemany("INSERT INTO book_history VALUES(?,?,?,?,?)", [(now, sym, s, p, 1.0) for s, p in walls])
        hist = c.execute("SELECT ts,side,price,qty FROM book_history WHERE symbol=? AND ts>? AND ts<?",
                         (sym, now - 900, now)).fetchall()
        # الجدران البعيدة عن السعر التي اختفت فقط تُحسب (لا تلك التي نُفّذت)
        hist = [h for h in hist if abs(h[2] / price - 1) > 0.002]
        sp = F.spoof_score(hist, walls)
    return {**kf, **bf, **tf, "spoof": sp}, price, atr


def decide(feats):
    """النموذج هو صاحب القرار؛ القواعد تشرح. بدون نموذج: قواعد فقط بعتبة عالية."""
    (sc, why), kind = detectors.score(feats)
    pl, ps = model.predict(feats); m = model.load() or {}
    tl, ts_ = m.get("thr_long"), m.get("thr_short")
    if pl is not None and tl and pl >= tl and (not config.LONG_NEEDS_BTC_UP or (feats.get("btc_trend") or 0) > 0):
        return "ACC", sc, pl, why if kind == "ACC" else ["نموذج: احتمال صعود مرتفع"]
    if ps is not None and ts_ and ps >= ts_:
        return "DIST", sc, ps, why if kind == "DIST" else ["نموذج: احتمال هبوط مرتفع"]
    if not m and sc >= config.ALERT_MIN_SCORE: return kind, sc, None, why
    return None, sc, (pl if kind == "ACC" else ps), why


def run_cycle(c, deriv, ref_raw):
    now = int(time.time())
    for sym in config.SYMBOLS:
        try:
            r = scan_symbol(c, sym, now, ref_raw)
            if not r: continue
            feats, price, atr = r; feats.update(deriv.get(sym, {}))
            kind, sc, prob, why = decide(feats)
            feats["score"] = sc
            db.save_snapshot(c, now, sym, price, {k: (v if v == v else None) for k, v in feats.items()})
            if not kind: continue
            last = c.execute("SELECT ts FROM signals WHERE symbol=? AND kind=? ORDER BY ts DESC LIMIT 1", (sym, kind)).fetchone()
            if not last or now - last[0] > 3600:
                c.execute("INSERT INTO signals(ts,symbol,kind,score,prob,price,atr,features,reasons) VALUES(?,?,?,?,?,?,?,?,?)",
                          (now, sym, kind, sc, prob, price, atr, json.dumps(feats, default=str),
                           json.dumps(why, ensure_ascii=False)))
            a = c.execute("SELECT ts FROM alerts WHERE symbol=? AND kind=?", (sym, kind)).fetchone()
            if not a or now - a[0] > config.COOLDOWN_MIN * 60:
                notifier.send(notifier.fmt(sym, kind, sc, prob, price, why, atr))
                c.execute("INSERT OR REPLACE INTO alerts VALUES(?,?,?)", (sym, kind, now))
        except Exception as e:
            print(f"[{sym}] error: {e!r}", file=sys.stderr)
    c.commit()


def need_retrain():
    if not os.path.exists(config.MODEL_PATH): return True
    return time.time() - os.path.getmtime(config.MODEL_PATH) > config.RETRAIN_HOURS * 3600


def retrain():
    t = time.time(); d = ml.dataset(config.SYMBOLS, config.TRAIN_BARS)
    if d.empty: print("train: no data"); return
    print("train:", model.train(d), "rows:", len(d), f"{time.time()-t:.0f}s")


def main():
    c = db.connect()
    print("labeled:", labeler.label_pending(c))
    if need_retrain():
        try: retrain()
        except Exception as e: print("train error:", repr(e), file=sys.stderr)
    deriv = {}
    for s in config.SYMBOLS:
        try: deriv[s] = sources.derivatives(s)
        except Exception: deriv[s] = {}
    for i in range(config.SAMPLES_PER_RUN):
        t0 = time.time(); ref_raw = sources.klines("BTCUSDT", "15m", 1000)
        run_cycle(c, deriv, ref_raw)
        if i < config.SAMPLES_PER_RUN - 1: time.sleep(max(0, config.SAMPLE_SECONDS - (time.time() - t0)))
    if time.gmtime().tm_hour == 0 and time.gmtime().tm_min < 10: notifier.send(notifier.daily(c))
    db.prune(c); c.close()


if __name__ == "__main__": main()
