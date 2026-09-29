import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_telegram_alert(symbol, signal_type, confidence, details):
    """إرسال إشعار منظم وجذاب على قناة تيليجرام"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ مفاتيح تيليجرام غير مضافة في البيئة.")
        return

    emoji = "🚀 [إشارة تجميع حيتان]" if signal_type == "BUY" else "⚠️ [إشارة تصريف حيتان]"
    action_text = "شراء / تجميع قبل القطيع" if signal_type == "BUY" else "جني أرباح / تخارج محتمل"
    
    clean_symbol = symbol.replace("/", "")

    message = f"""
{emoji}

• **العملة:** #{clean_symbol}
• **النوع:** `{action_text}`
• **نسبة ثقة AI:** `{confidence * 100:.1f}%`

📊 **تحليل السيولة وبوتات التلاعب:**
- **دفتر الأوامر:** `{details['ob_status']}`
- **حركة CVD:** `{details['cvd_status']}`
- **الأوامر المخفية (Iceberg):** `{details['iceberg_status']}`
- **تأكيد MTF (15m/1h/4h):** `{details['mtf_status']}`
- **تدفق On-Chain:** `{details['onchain_status']}`

💡 **التوصية:** {details['recommendation']}
"""

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message.strip(),
        "parse_mode": "Markdown"
    }

    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print(f"✅ تم إرسال تنبيه {symbol} بنجاح.")
        else:
            print(f"❌ فشل الإرسال: {res.text}")
    except Exception as e:
        print(f"خطأ اتصال بتيليجرام: {e}")
