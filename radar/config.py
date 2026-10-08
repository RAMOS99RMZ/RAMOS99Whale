import os
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,SUIUSDT,PEPEUSDT").split(",")
DB_PATH = os.getenv("DB_PATH", "data/radar.db")
MODEL_PATH = os.getenv("MODEL_PATH", "data/model.joblib")
BINANCE = "https://data-api.binance.vision/api/v3"
OKX = "https://www.okx.com/api/v5"
SAMPLE_SECONDS = int(os.getenv("SAMPLE_SECONDS", "60"))
SAMPLES_PER_RUN = int(os.getenv("SAMPLES_PER_RUN", "8"))
HORIZON_BARS = 16          # 16 شمعة 15m = 4 ساعات
HORIZON_MIN = HORIZON_BARS * 15
WARMUP = 900               # شموع إحماء (EMA800)
MIN_AUC = float(os.getenv("MIN_AUC", "0.53"))
P_LONG = float(os.getenv("P_LONG", "0.0"))   # 0 = يُختار تلقائياً من التدريب
TRAIN_BARS = int(os.getenv("TRAIN_BARS", "20000"))
FEE_RT = 0.0012            # رسوم+انزلاق ذهاباً وإياباً
TP_ATR, SL_ATR = 2.0, 1.0  # Triple barrier
ALERT_MIN_SCORE = float(os.getenv("ALERT_MIN_SCORE", "70"))
COOLDOWN_MIN = 120
MIN_LABELS_TO_TRAIN = 200
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
RETRAIN_HOURS = float(os.getenv("RETRAIN_HOURS", "6"))
LONG_NEEDS_BTC_UP = os.getenv("LONG_NEEDS_BTC_UP", "1") == "1"
