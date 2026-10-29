"""How well can sensors predict a fail? Balanced error rate (BER), two protocols.

  cv10        stratified 10-fold cross validation, comparable in spirit to the UCI page's baselines
              (they report BER about 33.5-40% with 40 selected features);
  time_split  train on the earliest 70% of runs, test on the latest 30%: what a fab would face,
              because the model is always applied to future lots.

Feature selection happens inside the pipeline, on training data only, so test folds never leak into it.
"""
import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(k=40):
    return make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(0.0), StandardScaler(),
                         SelectKBest(f_classif, k=k),
                         LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000))


def ber(y_true, y_pred):
    return 1.0 - balanced_accuracy_score(y_true, y_pred)


def evaluate(X, y, k=40, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    pred_cv = cross_val_predict(make_model(k), X, y, cv=cv)
    cut = int(len(y) * 0.7)
    model = make_model(k).fit(X[:cut], y[:cut])
    pred_t = model.predict(X[cut:])
    tp = int(((pred_t == 1) & (y[cut:] == 1)).sum())
    return {
        "k_features": k,
        "cv10_ber": round(ber(y, pred_cv), 3),
        "time_split_ber": round(ber(y[cut:], pred_t), 3),
        "time_split_test_runs": int(len(y) - cut),
        "time_split_test_fails": int(y[cut:].sum()),
        "time_split_fails_caught": tp,
        "time_split_false_alarms": int(((pred_t == 1) & (y[cut:] == 0)).sum()),
    }
