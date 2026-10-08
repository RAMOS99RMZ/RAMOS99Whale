"""قواعد شفافة تُنتج score 0-100 وأسباباً مقروءة. النموذج يحدد القرار، والقواعد تشرح "لماذا"."""


def score(f):
    g = lambda k: float(f.get(k) or 0) if f.get(k) == f.get(k) else 0.0
    acc = dist = 0.0; wa, wd = [], []
    if g("div_16") > 1.0: acc += 20; wa.append("CVD يصعد والسعر لا يتحرك (تجميع خفي)")
    if g("div_16") < -1.0: dist += 20; wd.append("CVD يهبط والسعر ثابت (تصريف خفي)")
    if g("div_48") > 1.0: acc += 10; wa.append("تباعد CVD على 12 ساعة")
    if g("div_48") < -1.0: dist += 10; wd.append("تباعد سلبي على 12 ساعة")
    if g("range_comp") < 0.35: acc += 5; dist += 5
    if g("big_buy") > 0.25: acc += 10; wa.append("صفقات حيتان كبيرة شراء")
    if g("big_sell") > 0.25: dist += 10; wd.append("صفقات حيتان كبيرة بيع")
    if g("absorption") > 0.4: acc += 10; wa.append("امتصاص بيع عند القاع")
    if g("supply") > 0.4: dist += 10; wd.append("عرض/تصريف عند القمة")
    if g("effort_result") > 1.5:
        if g("cvd_4") > 0: acc += 5; wa.append("حجم كبير دون حركة (امتصاص)")
        else: dist += 5; wd.append("حجم كبير دون حركة (توزيع)")
    if g("whale_net") > 0.1: acc += 10; wa.append("صافي شراء حيتان لحظي")
    if g("whale_net") < -0.1: dist += 10; wd.append("صافي بيع حيتان لحظي")
    if g("iceberg") > 0.15: wa.append("أوامر iceberg مخفية") if g("whale_net") >= 0 else wd.append("أوامر iceberg مخفية")
    if g("imb_1pct") > 0.3: acc += 5
    if g("imb_1pct") < -0.3: dist += 5
    if g("oi_chg") > 0.03 and g("funding") < 0: acc += 10; wa.append("OI يرتفع والتمويل سالب (ضغط شورت)")
    if g("oi_chg") > 0.03 and g("funding") > 0.0005: dist += 10; wd.append("رافعة شراء مفرطة")
    if g("btc_trend") > 0: acc += 5
    else: dist += 5
    if g("spoof") > 0.5: acc *= 0.8; dist *= 0.8; (wa if acc >= dist else wd).append("⚠️ جدران وهمية (spoofing)")
    if acc >= dist: return (min(acc, 100), wa), "ACC"
    return (min(dist, 100), wd), "DIST"
