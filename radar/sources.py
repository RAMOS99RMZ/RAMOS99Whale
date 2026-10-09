import requests, time
from . import config
S = requests.Session(); S.headers["User-Agent"] = "whale-radar-v2"
def _get(url, params=None, tries=3):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=10)
            if r.status_code == 200: return r.json()
            if r.status_code in (418, 429): time.sleep(5*(i+1))
        except requests.RequestException: time.sleep(1+i)
    return None
def klines(sym, interval="15m", limit=200, end=None):
    p = {"symbol": sym, "interval": interval, "limit": limit}
    if end: p["endTime"] = end
    return _get(f"{config.BINANCE}/klines", p) or []
def depth(sym, limit=500): return _get(f"{config.BINANCE}/depth", {"symbol": sym, "limit": limit})
def agg_trades(sym, start_ms):
    out, frm = [], start_ms
    for _ in range(10):
        d = _get(f"{config.BINANCE}/aggTrades", {"symbol": sym, "startTime": frm, "limit": 1000})
        if not d: break
        out += d
        if len(d) < 1000: break
        frm = d[-1]["T"] + 1
    return out
def okx_inst(sym): return sym.replace("USDT", "") + "-USDT-SWAP"
def derivatives(sym):
    inst = okx_inst(sym); res = {"funding": 0.0, "oi_chg": 0.0, "ls_ratio": 1.0}
    f = _get(f"{config.OKX}/public/funding-rate", {"instId": inst})
    if f and f.get("data"): res["funding"] = float(f["data"][0].get("fundingRate") or 0)
    oi = _get(f"{config.OKX}/rubik/stat/contracts/open-interest-volume", {"ccy": sym.replace("USDT",""), "period": "5m"})
    if oi and oi.get("data") and len(oi["data"]) > 12:
        now, prev = float(oi["data"][0][1]), float(oi["data"][12][1])
        res["oi_chg"] = (now - prev) / prev if prev else 0
    ls = _get(f"{config.OKX}/rubik/stat/contracts/long-short-account-ratio", {"ccy": sym.replace("USDT",""), "period": "5m"})
    if ls and ls.get("data"): res["ls_ratio"] = float(ls["data"][0][1])
    return res
