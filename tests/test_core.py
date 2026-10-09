from radar import labeler, detectors, features as F
def test_barrier_tp_sl():
    assert labeler.barrier("ACC", 100, 1, [103], [99.5], [102])[0] == 1
    assert labeler.barrier("ACC", 100, 1, [100.5], [98.9], [99])[0] == 0
    assert labeler.barrier("DIST", 100, 1, [100.5], [97.9], [98])[0] == 1
def test_score_accum():
    (s, why), k = detectors.score({"div_16": 1.5, "div_48": 1.2, "whale_net": .2, "absorption": .5, "btc_trend": 1})
    assert k == "ACC" and s >= 50 and why
def test_book():
    b = {"bids": [["99","10"],["98","1"]], "asks": [["101","1"]]}
    assert F.book_features(b, 100)["imb_1pct"] > 0
