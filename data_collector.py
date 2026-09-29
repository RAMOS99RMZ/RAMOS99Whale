import ccxt
import numpy as np
import pandas as pd
from config import TARGET_SYMBOLS, TIMEFRAMES

class WhaleDataCollector:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def fetch_orderbook_metrics(self, symbol):
        """رصد جدران البيع الشراء الوهمية والتوازن في دفتر الأوامر"""
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=100)
            bids = np.array(orderbook['bids'])
            asks = np.array(orderbook['asks'])

            if len(bids) == 0 or len(asks) == 0:
                return 0.0, 1.0, 1.0

            total_bid_vol = np.sum(bids[:, 1])
            total_ask_vol = np.sum(asks[:, 1])
            
            # اختلال دفتر الأوامر (Imbalance)
            imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-8)

            # رصد الجدران الوهمية (Spoofing Ratio)
            avg_bid = np.mean(bids[:, 1])
            avg_ask = np.mean(asks[:, 1])
            max_bid_wall = np.max(bids[:, 1]) / (avg_bid + 1e-8)
            max_ask_wall = np.max(asks[:, 1]) / (avg_ask + 1e-8)

            return float(imbalance), float(max_bid_wall), float(max_ask_wall)
        except Exception as e:
            print(f"خطأ سحب دفتر الأوامر لـ {symbol}: {e}")
            return 0.0, 1.0, 1.0

    def fetch_cvd_and_icebergs(self, symbol):
        """رصد الدلتا التراكمية CVD والأوامر المخفية (Iceberg Orders)"""
        try:
            trades = self.exchange.fetch_trades(symbol, limit=300)
            if not trades:
                return 0.0, 0.0

            df = pd.DataFrame(trades)
            buy_vol = df[df['side'] == 'buy']['amount'].sum()
            sell_vol = df[df['side'] == 'sell']['amount'].sum()
            total_vol = buy_vol + sell_vol + 1e-8

            cvd_ratio = (buy_vol - sell_vol) / total_vol

            # حساب مؤشر الأوامر المخفية: صفقات ضخمة في نطاق سعري ضيق جداً
            price_spread = (df['price'].max() - df['price'].min()) / (df['price'].mean() + 1e-8)
            iceberg_score = (total_vol / (price_spread + 1e-5)) / 100000.0

            return float(cvd_ratio), float(np.clip(iceberg_score, 0, 10))
        except Exception as e:
            print(f"خطأ حساب CVD لـ {symbol}: {e}")
            return 0.0, 0.0

    def fetch_mtf_momentum(self, symbol):
        """تحليل الاتجاه والزخم على الفريمات المختلفة (15m, 1h, 4h)"""
        mtf_signals = {}
        for tf in TIMEFRAMES:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=20)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # حساب تغير السعر مع الحجم
                price_change = (df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0]
                vol_momentum = df['volume'].iloc[-1] / (df['volume'].mean() + 1e-8)
                
                mtf_signals[tf] = {
                    'price_change': float(price_change),
                    'vol_momentum': float(vol_momentum)
                }
            except Exception:
                mtf_signals[tf] = {'price_change': 0.0, 'vol_momentum': 1.0}
        return mtf_signals

    def get_onchain_flow_proxy(self, symbol):
        """مؤشر تدفقات On-Chain لتقدير حركة المحافظ الكبرى خروجاً ودخولاً من المنصات"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            base_volume = ticker.get('baseVolume', 1.0)
            quote_volume = ticker.get('quoteVolume', 1.0)
            
            # نسبة ضغط السيولة
            flow_index = np.sin(quote_volume / (base_volume + 1e-8))
            return float(flow_index)
        except Exception:
            return 0.0
