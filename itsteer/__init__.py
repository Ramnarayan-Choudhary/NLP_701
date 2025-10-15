# itsteer/__init__.py
from .steering import SteeringContext
from .eval_runner import evaluate_model
from .eval_metrics import accuracy, content_effect_metrics

def compute_metrics(y_true, y_pred, plaus):
    """Convenience wrapper that returns accuracy + CE metrics in one dict."""
    mets = content_effect_metrics(y_true, y_pred, plaus)
    mets["accuracy"] = accuracy(y_true, y_pred)
    return mets