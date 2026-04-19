# -*- coding: utf-8 -*-
"""Shared pytest fixtures for the classification-tool test suite."""
from __future__ import annotations

import importlib.util as _ilu
import pathlib
import sys
from typing import List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Tk stubbing — install at collection time so tests that import the UI
# mixins (app_cluster, app_train, app_apply) can load on headless runners.
# Only stubs what is actually missing; real tkinter is left alone when
# available. See tests/test_app_mixins_import_smoke.py for the rationale.
# ---------------------------------------------------------------------------

def _make_ui_mock(name: str) -> MagicMock:
    m = MagicMock()
    m.__name__ = name
    m.__spec__ = MagicMock()
    return m


for _mod in (
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "ui_theme",
    "ui_widgets",
    "customtkinter",
):
    if _mod not in sys.modules and _ilu.find_spec(_mod.split(".")[0]) is None:
        sys.modules[_mod] = _make_ui_mock(_mod)


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


# ---------------------------------------------------------------------------
# Tiny synthetic datasets — для быстрых ml_training-тестов (без UI/IO)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_balanced_dataset():
    """30 примеров, 2 класса по 15. Достаточно для train_model + holdout."""
    texts = [f"text example number {i} about topic" for i in range(30)]
    labels = [("a" if i % 2 == 0 else "b") for i in range(30)]
    return texts, labels


@pytest.fixture(scope="session")
def tagged_balanced_dataset():
    """То же, но с [DESC]/[CLIENT]/[OPERATOR] секциями для field_dropout."""
    sample = (
        "[DESC]\nописание транзакции\n"
        "[CLIENT]\nклиент говорит\n"
        "[OPERATOR]\nоператор отвечает"
    )
    texts = [sample] * 30
    labels = [("a" if i % 2 == 0 else "b") for i in range(30)]
    return texts, labels


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
