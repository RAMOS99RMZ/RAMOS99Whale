import os

# الأصول المستهدفة للتتبع
TARGET_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "LINK/USDT",
    "ONDO/USDT",
    "AVAX/USDT"
]

# الفريمات الزمنية المعتمدة للتحليل متعدد الفريمات (MTF)
TIMEFRAMES = ["15m", "1h", "4h"]

# مفاتيح تيليجرام من البيئة المشفرة
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# عتبات كشف تلاعب بوتات الحيتان (2026 Bot Thresholds)
SPOOFING_THRESHOLD = 3.5      # جدار وهمي أكبر من متوسط عمق الدفتر بـ 3.5 ضعف
CVD_DIVERGENCE_RATIO = 0.35    # انحراف الفارق بين الشراء والبيع الماركت
ICEBERG_VOLUME_RATIO = 2.5     # حجم صفقات مخفية مقارنة بالنطاق السعري
CONFIDENCE_MIN = 0.65          # حد الأدنى لثقة الذكاء الاصطناعي لإرسال التنبيه

# مسار حفظ ذاكرة النموذج للتعلّم المستمر
MODEL_STATE_FILE = "model_state.json"
