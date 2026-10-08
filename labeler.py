"""Triple-Barrier: TP=2ATR، SL=1ATR، أو انتهاء المهلة. يُطبّق على الإشارات بعد مرور HORIZON."""
import time
from . import config, sources
def barrier(kind, entry, atr, highs, lows, closes):
    tp = entry + (config.TP_ATR*atr if kind == "ACC" else -config.TP_ATR*atr)
    sl = entry - (config.SL_ATR*atr if kind == "ACC" else -config.SL_ATR*atr)
    for h, l in zip(highs, lows):
        if kind == "ACC":
            if l <= sl: return 0, -config.SL_ATR*atr/entry
            if h >= tp: return 1, config.TP_ATR*atr/entry
        else:
            if h >= sl: return 0, -config.SL_ATR*atr/entry
            if l <= tp: return 1, config.TP_ATR*atr/entry
    r = (closes[-1]-entry)/entry * (1 if kind == "ACC" else -1) if closes else 0
    return int(r > 0), r
def label_pending(c):
    cut = int(time.time()) - config.HORIZON_MIN*60; n = 0
    for sid, ts, sym, kind, price, atr in c.execute(
        "SELECT id,ts,symbol,kind,price,atr FROM signals WHERE label IS NULL AND ts<?", (cut,)).fetchall():
        k = sources._get(f"{config.BINANCE}/klines", {"symbol": sym, "interval": "5m", "startTime": ts*1000,
                         "limit": config.HORIZON_MIN//5})
        if not k: continue
        lab, ret = barrier(kind, price, atr, [float(x[2]) for x in k], [float(x[3]) for x in k], [float(x[4]) for x in k])
        c.execute("UPDATE signals SET label=?,ret=?,labeled_ts=? WHERE id=?", (lab, ret, int(time.time()), sid)); n += 1
    c.commit(); return n
