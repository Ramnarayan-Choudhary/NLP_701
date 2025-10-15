def test_import():
    import itsteer
def test_metrics_shapes():
    from itsteer.eval_metrics import accuracy, content_effect_metrics
    y_true = [True, False, True, False]
    y_pred = [True, True, False, False]
    plaus = [True, True, False, False]
    acc = accuracy(y_true, y_pred)
    mets = content_effect_metrics(y_true, y_pred, plaus)
    assert isinstance(acc, float)
    assert "total_content_effect" in mets
