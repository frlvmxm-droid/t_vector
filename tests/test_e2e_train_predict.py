# -*- coding: utf-8 -*-
"""End-to-end integration test: train → persist → predict.

Uses only stdlib + scikit-learn (no tkinter, no torch, no SBERT).
Verifies the full pipeline works without requiring a real dataset.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Synthetic dataset — 3 classes, 12 examples each (36 total)
# ---------------------------------------------------------------------------
_LABELS = ["задержка_платежа", "блокировка_карты", "смена_пароля"]
_TEXTS = (
    # задержка_платежа
    [
        "Почему платёж не прошёл? Жду третий день.",
        "Перевод завис, деньги не поступили на счёт.",
        "Задержка зачисления денег на карту.",
        "Не пришли деньги на счёт после перевода.",
        "Уже два дня жду платёж — не дошёл.",
        "Когда будет зачислен перевод на карту Сбербанка?",
        "Платёж завис в обработке, статус — выполняется.",
        "Деньги списались но не поступили получателю.",
        "Задержка перевода — прошло больше суток.",
        "Не могу понять где моя транзакция, деньги списаны.",
        "Платёж висит в обработке уже 48 часов.",
        "Зачисление платежа задерживается больше дня.",
    ]
    +
    # блокировка_карты
    [
        "Карта заблокирована, оплатить не могу.",
        "Заблокировали карту без предупреждения.",
        "При попытке оплаты карта была заблокирована.",
        "Карту заблокировали, нужно разблокировать.",
        "Почему заблокировали мою дебетовую карту?",
        "Не могу снять наличные — карта заблокирована.",
        "Терминал отклонил карту — она заблокирована.",
        "Карта не работает — банк заблокировал её.",
        "Как разблокировать карту после блокировки?",
        "Оплата не прошла, говорят карта заблокирована.",
        "Карту заблокировали при оплате в интернете.",
        "После трёх неверных ПИН-кодов карта заблокирована.",
    ]
    +
    # смена_пароля
    [
        "Хочу изменить пароль от личного кабинета.",
        "Как сменить пароль в мобильном приложении?",
        "Забыл пароль от интернет-банка, нужна помощь.",
        "Нужно сбросить пароль от онлайн-банка.",
        "Как поменять ПИН-код карты?",
        "Смена пароля в приложении — куда нажать?",
        "Не помню пароль, как восстановить доступ?",
        "Хочу сменить пин-код на новый.",
        "Как изменить секретный код от карты?",
        "Приложение не пускает — забыл пароль.",
        "Сброс пароля — не приходит смс с кодом.",
        "Поменял пароль и теперь не могу войти.",
    ]
)
_LABELS_FULL = _LABELS[0:1] * 12 + _LABELS[1:2] * 12 + _LABELS[2:3] * 12


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    """Train a minimal TF-IDF model and return (bundle_path, eval_metrics)."""
    from ml_vectorizers import make_hybrid_vectorizer
    from ml_training import train_model
    from experiment_log import log_experiment

    tmp_path = tmp_path_factory.mktemp("model")

    features = make_hybrid_vectorizer(
        char_ng=(2, 4),
        word_ng=(1, 1),
        min_df=1,
        max_features=5000,
        use_svd=False,
        use_lemma=False,
        use_per_field=False,
        use_meta=False,
        sublinear_tf=True,
    )

    pipe, clf_type, report, labels, cm, extras = train_model(
        X=_TEXTS,
        y=_LABELS_FULL,
        features=features,
        C=1.0,
        max_iter=500,
        balanced=True,
        calib_method="sigmoid",
        test_size=0.25,
        random_state=42,
        use_smote=False,
    )

    assert pipe is not None
    assert labels is not None
    assert len(labels) == 3

    rep_dict = extras.get("report_dict") or {}
    eval_metrics = {
        "macro_f1":  round(float((rep_dict.get("macro avg") or {}).get("f1-score", 0.0)), 4),
        "accuracy":  round(float(rep_dict.get("accuracy", 0.0)), 4),
        "n_train":   extras.get("n_train", 0),
        "n_test":    extras.get("n_test", 0),
        "per_class_f1": {},
        "trained_at": "2024-01-01T00:00:00",
    }

    import joblib
    bundle_path = tmp_path / "test_model.joblib"
    bundle = {
        "artifact_type": "train_model_bundle",
        "pipeline": pipe,
        "schema_version": 1,
        "per_class_thresholds": {},
        "eval_metrics": eval_metrics,
    }
    joblib.dump(bundle, bundle_path, compress=1)

    # Smoke-test experiment log (should not raise)
    log_experiment(str(bundle_path), {"train_mode": "tfidf", "C": 1.0}, eval_metrics)

    return bundle_path, eval_metrics


def test_train_produces_nonzero_f1(trained_bundle):
    """Trained model should achieve macro F1 > 0 on synthetic data."""
    _, eval_metrics = trained_bundle
    assert eval_metrics["macro_f1"] > 0.0, "macro F1 должен быть > 0"


def test_n_train_n_test_recorded(trained_bundle):
    """n_train and n_test should be positive integers."""
    _, eval_metrics = trained_bundle
    assert eval_metrics["n_train"] > 0
    assert eval_metrics["n_test"] > 0
    assert eval_metrics["n_train"] + eval_metrics["n_test"] == len(_TEXTS)


def test_bundle_loadable_and_predicts(trained_bundle):
    """Saved bundle should load and produce valid predictions."""
    import joblib

    bundle_path, _ = trained_bundle
    bundle = joblib.load(bundle_path)

    pipe = bundle["pipeline"]
    preds = pipe.predict(_TEXTS[:6])

    assert len(preds) == 6
    valid_labels = set(_LABELS)
    for p in preds:
        assert p in valid_labels, f"Предсказан неизвестный класс: {p!r}"


def test_bundle_has_eval_metrics(trained_bundle):
    """Bundle dict must contain eval_metrics with required keys."""
    import joblib

    bundle_path, _ = trained_bundle
    bundle = joblib.load(bundle_path)

    m = bundle.get("eval_metrics")
    assert m is not None, "eval_metrics отсутствует в бандле"
    assert "macro_f1" in m
    assert "accuracy" in m
    assert "n_train" in m
    assert "n_test" in m


def test_predict_proba_available(trained_bundle):
    """Pipeline should support predict_proba (CalibratedClassifierCV)."""
    import joblib
    import numpy as np

    bundle_path, _ = trained_bundle
    bundle = joblib.load(bundle_path)
    pipe = bundle["pipeline"]

    proba = pipe.predict_proba(_TEXTS[:3])
    assert proba.shape == (3, 3), f"Ожидали (3, 3), получили {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
