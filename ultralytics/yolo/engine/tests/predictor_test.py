from ultralytics.yolo.engine import predictor

bp = predictor.BasePredictor()
    
def test_specificResult():
    # [ボトル青, ボトルオレンジ, カテーテル, ガイドワイヤー, 骨髄, 腰椎]
    test_list_0 = [0, 0, 0, 0, 0, 1] # youtui
    test_list_1 = [0, 0, 0, 0, 1, 0] # kotuzui
    test_list_2 = [0, 0, 0, 1, 0, 0] # unknown
    test_list_3 = [0, 0, 1, 0, 0, 0] # unknown
    test_list_4 = [0, 1, 0, 0, 0, 0] # unknown
    test_list_5 = [1, 0, 0, 0, 0, 0] # unknown
    test_list_6 = [1, 1, 0, 0, 0, 0] # blood
    test_list_7 = [0, 0, 1, 1, 0, 0] # catheter
    test_list_8 = [1, 0, 1, 1, 0, 0] # unknown
    test_list_9 = [1, 0, 2, 2, 0, 0] # catheter
    test_list_10 = [1, 0, 2, 1, 0, 0] # unknown
    test_list_11 = [1, 0, 2, 3, 1, 0] # catheter
    test_list_12 = [1, 2, 4, 3, 0, 4] # unknown
    test_list_13 = [0, 0, 0, 0, 0, 0] # unknown
    test_list_14 = [1, 2, 3, 3, 0, 4] # youtui
    test_list_15 = [3, 3, 2, 2, 0, 2] # blood
    test_list_16 = [4, 3, 2, 1, 1, 2] # blood



    assert bp.specificResult(test_list_0) == "youtui"
    assert bp.specificResult(test_list_1) == "kotuzui"
    assert bp.specificResult(test_list_2) == "unknown"
    assert bp.specificResult(test_list_3) == "unknown"
    assert bp.specificResult(test_list_4) == "unknown"
    assert bp.specificResult(test_list_5) == "unknown"
    assert bp.specificResult(test_list_6) == "blood"
    assert bp.specificResult(test_list_7) == "catheter"
    assert bp.specificResult(test_list_8) == "unknown"
    assert bp.specificResult(test_list_9) == "catheter"
    assert bp.specificResult(test_list_10) == "unknown"
    assert bp.specificResult(test_list_11) == "catheter"
    assert bp.specificResult(test_list_12) == "unknown"
    assert bp.specificResult(test_list_13) == "unknown"
    assert bp.specificResult(test_list_14) == "youtui"
    assert bp.specificResult(test_list_15) == "blood"
    assert bp.specificResult(test_list_16) == "blood"

    

