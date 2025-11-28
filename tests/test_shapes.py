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


def test_stratified_split_examples():
    from collections import Counter
    from itsteer.data import Example, stratified_split_examples

    examples = [
        Example(id="a", syllogism="s1", validity=True, plausibility=True),
        Example(id="b", syllogism="s2", validity=True, plausibility=False),
        Example(id="c", syllogism="s3", validity=False, plausibility=True),
        Example(id="d", syllogism="s4", validity=False, plausibility=False),
        Example(id="e", syllogism="s5", validity=False, plausibility=False),
    ]

    train, test = stratified_split_examples(examples, train_ratio=0.5, seed=123)

    def counts(xs):
        return Counter((bool(ex.validity), bool(ex.plausibility)) for ex in xs)

    # All examples should be accounted for across the splits
    assert len(train) + len(test) == len(examples)
    assert counts(train) + counts(test) == counts(examples)

    # Deterministic given the same seed
    train2, test2 = stratified_split_examples(examples, train_ratio=0.5, seed=123)
    assert [ex.id for ex in train] == [ex.id for ex in train2]
    assert [ex.id for ex in test] == [ex.id for ex in test2]
