import requests
from . import config
def send(text):
    if not (config.TG_TOKEN and config.TG_CHAT): print(text); return
    try: requests.post(f"https://api.telegram.org/bot{config.TG_TOKEN}/sendMessage",
                       json={"chat_id": config.TG_CHAT, "text": text, "parse_mode": "HTML"}, timeout=10)
    except requests.RequestException as e: print("telegram error", e)
def fmt(sym, kind, score, prob, price, reasons, atr):
    tag = "🟢 تجميع حيتان" if kind == "ACC" else "🔴 تصريف حيتان"
    p = f"{prob*100:.0f}%" if prob is not None else "غير مدرب بعد"
    tp = price + 2*atr if kind == "ACC" else price - 2*atr; sl = price - atr if kind == "ACC" else price + atr
    return (f"<b>{tag} — {sym}</b>\nالسعر: {price:.6g}\nالقوة: {score:.0f}/100 | احتمال النجاح: {p}\n"
            f"الهدف: {tp:.6g} | الوقف: {sl:.6g}\nالأسباب: " + "، ".join(reasons))


def daily(c):
    """تقرير يومي: أداء الإشارات الحقيقي آخر 7 أيام."""
    import time
    rows = c.execute("SELECT kind, COUNT(*), AVG(label), AVG(ret) FROM signals WHERE label IS NOT NULL AND ts>? GROUP BY kind",
                     (int(time.time()) - 7 * 86400,)).fetchall()
    if not rows: return "📊 تقرير يومي: لا توجد إشارات موسومة بعد."
    lines = [f"{'تجميع' if k == 'ACC' else 'تصريف'}: {n} إشارة | نجاح {w*100:.0f}% | متوسط {r*100:.2f}%" for k, n, w, r in rows]
    return "📊 <b>أداء الرادار (7 أيام)</b>\n" + "\n".join(lines)
