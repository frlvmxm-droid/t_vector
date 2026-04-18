# -*- coding: utf-8 -*-
"""Shared pytest fixtures for the classification-tool test suite."""
from __future__ import annotations

import pathlib
import sys
from typing import List

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Canonical synthetic dataset — 3 Russian banking classes, 12 examples each
# ---------------------------------------------------------------------------

SAMPLE_LABELS: List[str] = ["задержка_платежа", "блокировка_карты", "смена_пароля"]

SAMPLE_TEXTS: List[str] = (
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
    + [
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
    + [
        "Хочу сменить пароль от личного кабинета.",
        "Как изменить пин-код карты?",
        "Не помню пароль — не могу войти.",
        "Сменить пароль через приложение не получается.",
        "Пин-код не подходит — что делать?",
        "Хочу поменять пароль от мобильного банка.",
        "Восстановление доступа — забыл пароль.",
        "Нужно обновить пин-код карты, не знаю как.",
        "Пин-код заблокирован после ввода неправильного.",
        "Как поменять пароль в интернет-банке?",
        "Не могу изменить пин-код через банкомат.",
        "Смена пароля личного кабинета онлайн.",
    ]
)

SAMPLE_Y: List[str] = [lbl for lbl in SAMPLE_LABELS for _ in range(12)]

# Structured dialog format used by PerFieldVectorizer tests
STRUCTURED_TEXTS: List[str] = [
    "[CHANNEL]\ncall\n[DESC]\nклиент хочет закрыть счёт\n[CLIENT]\nхочу закрыть счёт\n[OPERATOR]\nхорошо понял",
    "[CHANNEL]\nchat\n[CLIENT]\nпомогите с картой\n[OPERATOR]\nкакой тип карты",
    "[CHANNEL]\ncall\n[DESC]\nвопрос по кредиту\n[CLIENT]\nкак погасить кредит досрочно\n[OPERATOR]\nрасскажу порядок",
    "[CHANNEL]\nchat\n[DESC]\nперевод не прошёл\n[CLIENT]\nденьги не дошли\n[OPERATOR]\nпроверяю статус",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_texts() -> List[str]:
    return SAMPLE_TEXTS


@pytest.fixture(scope="session")
def sample_labels() -> List[str]:
    return SAMPLE_LABELS


@pytest.fixture(scope="session")
def sample_y() -> List[str]:
    return SAMPLE_Y


@pytest.fixture(scope="session")
def structured_texts() -> List[str]:
    return STRUCTURED_TEXTS


@pytest.fixture(scope="session")
def minimal_bundle(tmp_path_factory):
    """Minimal sklearn TF-IDF + LinearSVC bundle for apply/predict tests."""
    import joblib
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    pipe = Pipeline([
        ("vec", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)),
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=200), cv=3)),
    ])
    pipe.fit(SAMPLE_TEXTS, SAMPLE_Y)

    bundle = {
        "artifact_type": "train_model_bundle",
        "schema_version": 1,
        "pipeline": pipe,
        "config": {"train_mode": "tfidf", "C": 1.0},
        "per_class_thresholds": {},
        "eval_metrics": {
            "macro_f1": 0.9,
            "accuracy": 0.9,
            "n_train": len(SAMPLE_TEXTS),
            "n_test": 0,
        },
        "class_examples": {lbl: [] for lbl in SAMPLE_LABELS},
    }

    tmp = tmp_path_factory.mktemp("bundle")
    path = tmp / "test_model.joblib"
    joblib.dump(bundle, path, compress=1)
    return path, bundle
