from scipy import stats
import numpy as np
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, classification_report
import joblib

def run_cv_perms(estimator, matrices, metadata, cv):
    y = metadata.cluster_label.values.astype(int)
    scores = []
    maps = []

    for train_idx, test_idx in cv.split(matrices, y, groups=metadata.CEN_ANOM.values):
        X_train, y_train = matrices[train_idx], y[train_idx]
        X_test, y_test = matrices[test_idx], y[test_idx]
        estimator.fit(X_train, y_train)

        y_pred = estimator.predict(X_test)

        scores.append(
            f1_score(y_test, y_pred, average="macro")
        )

        reg = estimator.named_steps["classifier"]
        
        # This should be moved outisde the loop
        masker = estimator.named_steps["matrixmasker"]

        # Compute Haufe's transform to make coefs interpretable
        X = masker.transform(matrices)
        sigma_X = np.cov(X.T)
        W = reg.coef_.T
        patterns = sigma_X @ W

        maps.append(patterns)
    
    weights = np.stack(maps, axis=0)
    return scores, weights

# TODO Joblib that, I suppose there should be some very similar code in cross validate
# TODO Allow passing index to shuffle
def partial_f1_func(y_test, y_pred):
    return f1_score(y_test, y_pred, average="macro")

def run_cv(estimator, matrices, metadata, cv, score_func=partial_f1_func):
    y = metadata.cluster_label.values.astype(int)
    n_classes = len(np.unique(y))
    scores = []
    maps = []
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i, (train_idx, test_idx) in enumerate(cv.split(matrices, y, groups=metadata.CEN_ANOM.values)):
        X_train, y_train = matrices[train_idx], y[train_idx]
        X_test, y_test = matrices[test_idx], y[test_idx]
        estimator.fit(X_train, y_train)

        y_pred = estimator.predict(X_test)
        print(classification_report(y_test, y_pred))
        cm += confusion_matrix(y_test, y_pred, labels=range(n_classes))

        scores.append(
            score_func(y_test, y_pred)
        )

        reg = estimator.named_steps["classifier"]
        
        # This should be moved outisde the loop
        masker = estimator.named_steps["matrixmasker"]

        # Compute Haufe's transform to make coefs interpretable
        X = masker.transform(matrices)
        sigma_X = np.cov(X.T)
        W = reg.coef_.T
        patterns = sigma_X @ W

        maps.append(patterns)
        joblib.dump(estimator, f"output/estimator_{i}.joblib")
    
    hmat = np.stack(maps, axis=0)
    return scores, cm, hmat

macro_f1 = make_scorer(
    f1_score, average="macro"
)
