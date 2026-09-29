from config import TARGET_SYMBOLS, CONFIDENCE_MIN
from data_collector import WhaleDataCollector
from ai_engine import SelfLearningWhaleAI
from telegram_bot import send_telegram_alert

def main():
    collector = WhaleDataCollector()
    ai = SelfLearningWhaleAI()

    print("⚡ [Whale Shield 2026] بدء فحص حركة الحيتان والبوتات...")

    for symbol in TARGET_SYMBOLS:
        print(f"🔍 جاري فحص: {symbol}")
        
        # 1. جمع البيانات
        ob_imb, bid_wall, ask_wall = collector.fetch_orderbook_metrics(symbol)
        cvd, iceberg = collector.fetch_cvd_and_icebergs(symbol)
        mtf_data = collector.fetch_mtf_momentum(symbol)
        onchain_flow = collector.get_onchain_flow_proxy(symbol)

        # 2. تحضير المتجهات وتحليل الذكاء الاصطناعي
        features = ai.extract_feature_vector(ob_imb, bid_wall, ask_wall, cvd, iceberg, mtf_data, onchain_flow)
        signal, confidence = ai.analyze_whale_intent(features)

        # 3. إرسال التنبيهات في حال توفر شروط الثقة
        if signal != 0 and confidence >= CONFIDENCE_MIN:
            signal_type = "BUY" if signal == 1 else "SELL"
            
            # تجهيز التفاصيل لتنسيق الرسالة
            details = {
                'ob_status': f"جدار بيع وهمي ({ask_wall:.1f}x)" if signal == 1 else f"جدار شراء وهمي ({bid_wall:.1f}x)",
                'cvd_status': "امتصاص شراء ماركت" if cvd > 0 else "ضغط بيع متخفي",
                'iceberg_status': "رصد تنفيذ أوامر مخفية" if iceberg > 2.0 else "طبيعي",
                'mtf_status': "توافق زخم صاعد" if mtf_data['1h']['price_change'] > 0 else "توافق هابط",
                'onchain_status': "تدفقات سحب من المنصات" if onchain_flow > 0 else "تدفق إيداع",
                'recommendation': "تجميع تدريجي بالقرب من الدعم." if signal == 1 else "تخفيف الكميات وحماية الأرباح."
            }

            send_telegram_alert(symbol, signal_type, confidence, details)
            
            # تغذية نموذج التعلم الآلي
            ai.learn_and_update(features, signal)

    print("🏁 اكتملت دورة الفحص وتحديث ذاكرة AI.")

if __name__ == "__main__":
    main()
