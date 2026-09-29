import json
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from config import MODEL_STATE_FILE, CONFIDENCE_MIN

class SelfLearningWhaleAI:
    def __init__(self):
        # نموذج تعلم آلي مستمر
        self.model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
        self.classes = np.array([-1, 0, 1]) # -1: تصريف/بيع، 0: محايد، 1: تجميع/شراء
        self.is_initialized = False
        self.load_model_state()

    def load_model_state(self):
        """تحميل أوزان النموذج السابقة لاستمرار التعلم الذاتي"""
        if os.path.exists(MODEL_STATE_FILE):
            try:
                with open(MODEL_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    coef = np.array(data['coef'])
                    intercept = np.array(data['intercept'])
                    self.model.coef_ = coef
                    self.model.intercept_ = intercept
                    self.model.classes_ = self.classes
                    self.is_initialized = True
                    print("🧠 [AI Engine] تم تحميل ذاكرة الذكاء الاصطناعي بنجاح.")
            except Exception as e:
                print(f"⚠️ تعذر تحميل الذاكرة، سيتم استخدام التحديد الخوارزمي: {e}")
        else:
            print("🌱 [AI Engine] إنشاء نموذج ذكاء اصطناعي جديد...")

    def save_model_state(self):
        """حفظ الأوزان المحدثة في ملف JSON للتكامل مع GitHub Actions"""
        if hasattr(self.model, 'coef_'):
            data = {
                'coef': self.model.coef_.tolist(),
                'intercept': self.model.intercept_.tolist()
            }
            with open(MODEL_STATE_FILE, 'w') as f:
                json.dump(data, f, indent=4)
            print("💾 [AI Engine] تم حفظ التحديثات الذاتية للنموذج بنجاح.")

    def extract_feature_vector(self, ob_imb, bid_wall, ask_wall, cvd, iceberg, mtf_data, onchain_flow):
        """دمج كافة العوامل في مصفوفة واحدة متطورة"""
        tf_15m = mtf_data.get('15m', {'price_change': 0, 'vol_momentum': 1})
        tf_1h = mtf_data.get('1h', {'price_change': 0, 'vol_momentum': 1})
        tf_4h = mtf_data.get('4h', {'price_change': 0, 'vol_momentum': 1})

        features = [
            ob_imb,
            bid_wall,
            ask_wall,
            cvd,
            iceberg,
            tf_15m['price_change'],
            tf_15m['vol_momentum'],
            tf_1h['price_change'],
            tf_1h['vol_momentum'],
            tf_4h['price_change'],
            tf_4h['vol_momentum'],
            onchain_flow
        ]
        return np.array([features])

    def analyze_whale_intent(self, features):
        """توقع نية الحيتان مع نسبة الثقة"""
        ob_imb = features[0][0]
        bid_wall = features[0][1]
        ask_wall = features[0][2]
        cvd = features[0][3]
        iceberg = features[0][4]

        # قواعد رصد سلوك بوتات الحيتان (Rule Engine)
        heuristic_signal = 0
        confidence = 0.50

        # تجميع: جدار بيع وهمي (Spoofing) + شراء ماركت شرس (CVD مثبت)
        if ask_wall > 3.0 and cvd > 0.20:
            heuristic_signal = 1
            confidence = 0.85
        # تصريف: جدار شراء وهمي + بيع ماركت متخفي
        elif bid_wall > 3.0 and cvd < -0.20:
            heuristic_signal = -1
            confidence = 0.85
        # امتصاص بأوامر مخفية (Iceberg Absorption)
        elif iceberg > 3.0 and abs(cvd) > 0.3:
            heuristic_signal = 1 if cvd > 0 else -1
            confidence = 0.78

        # دمج مع نموذج التعلم الآلي إذا كان مهيأً
        if self.is_initialized:
            try:
                probs = self.model.predict_proba(features)[0]
                ai_pred = self.classes[np.argmax(probs)]
                ai_conf = np.max(probs)

                # تحديث الإشارة بالذكاء الاصطناعي
                if ai_conf > CONFIDENCE_MIN:
                    return ai_pred, float(ai_conf)
            except Exception:
                pass

        return heuristic_signal, float(confidence)

    def learn_and_update(self, features, actual_target):
        """تحديث الأوزان أونلاين بناءً على حركة السعر الفعلية"""
        try:
            self.model.partial_fit(features, [actual_target], classes=self.classes)
            self.is_initialized = True
            self.save_model_state()
        except Exception as e:
            print(f"خطأ تحديث التعلم: {e}")
