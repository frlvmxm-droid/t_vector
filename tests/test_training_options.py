# -*- coding: utf-8 -*-
"""Tests for the TrainingOptions dataclass и legacy-shim в train_model().

Покрывает:
  1. Дефолтный конструктор строит валидный объект.
  2. Передача legacy kwargs → DeprecationWarning + TrainingOptions собран.
  3. Одновременная передача options и legacy kwargs → TypeError.
  4. Неизвестный kwarg → TypeError.
  5. frozen: попытка мутации поля → FrozenInstanceError.
  6. slots: __dict__ отсутствует, посторонние атрибуты не принимаются.
  7. Конфигурация поле-по-полю корректно достигает train_model.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from ml_training import TrainingOptions, train_model


def _simple_vectorizer() -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 1), max_features=200))])


def _tiny_dataset():
    rng = np.random.default_rng(0)
    texts = [f"text example number {i} about topic" for i in range(30)]
    labels = [("a" if i % 2 == 0 else "b") for i in range(30)]
    rng.shuffle(texts)
    return texts, labels


class TestTrainingOptionsDataclass:
    def test_default_construction(self):
        opts = TrainingOptions()
        assert opts.calib_method == "sigmoid"
        assert opts.use_smote is True
        assert opts.oversample_strategy == "cap"
        assert opts.max_dup_per_sample == 5
        assert opts.run_cv is False
        assert opts.use_field_dropout is False
        assert opts.use_fuzzy_dedup is False

    def test_frozen_instance(self):
        opts = TrainingOptions()
        with pytest.raises(dataclasses.FrozenInstanceError):
            opts.use_smote = False  # type: ignore[misc]

    def test_slots_declared(self):
        opts = TrainingOptions()
        # slots=True → нет __dict__ и __slots__ содержит поля.
        assert not hasattr(opts, "__dict__")
        assert hasattr(TrainingOptions, "__slots__")
        assert "use_smote" in TrainingOptions.__slots__

    def test_custom_values_preserved(self):
        opts = TrainingOptions(
            use_smote=False,
            use_field_dropout=True,
            field_dropout_prob=0.3,
            field_dropout_copies=4,
            use_fuzzy_dedup=True,
            fuzzy_dedup_threshold=90,
        )
        assert opts.use_smote is False
        assert opts.use_field_dropout is True
        assert opts.field_dropout_prob == 0.3
        assert opts.field_dropout_copies == 4
        assert opts.use_fuzzy_dedup is True
        assert opts.fuzzy_dedup_threshold == 90


class TestTrainModelShim:
    def test_new_api_no_warning(self):
        X, y = _tiny_dataset()
        opts = TrainingOptions(use_smote=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipe, _, _, _, _, _ = train_model(
                X, y, _simple_vectorizer(),
                C=1.0, max_iter=200, balanced=True,
                test_size=0.2, random_state=42,
                options=opts,
            )
        assert pipe is not None

    def test_legacy_kwargs_emit_deprecation_warning(self):
        X, y = _tiny_dataset()
        with pytest.warns(DeprecationWarning, match="use_smote"):
            pipe, _, _, _, _, _ = train_model(
                X, y, _simple_vectorizer(),
                C=1.0, max_iter=200, balanced=True,
                test_size=0.2, random_state=42,
                use_smote=False,
            )
        assert pipe is not None

    def test_options_and_legacy_kwargs_conflict(self):
        X, y = _tiny_dataset()
        opts = TrainingOptions()
        with pytest.raises(TypeError, match="одновременно"):
            train_model(
                X, y, _simple_vectorizer(),
                C=1.0, max_iter=200, balanced=True,
                test_size=0.2, random_state=42,
                options=opts,
                use_smote=False,
            )

    def test_unknown_kwarg_rejected(self):
        X, y = _tiny_dataset()
        with pytest.raises(TypeError, match="неизвестные"):
            train_model(
                X, y, _simple_vectorizer(),
                C=1.0, max_iter=200, balanced=True,
                test_size=0.2, random_state=42,
                not_a_real_flag=True,
            )

    def test_default_options_no_warning(self):
        X, y = _tiny_dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipe, _, _, _, _, _ = train_model(
                X, y, _simple_vectorizer(),
                C=1.0, max_iter=200, balanced=True,
                test_size=0.2, random_state=42,
            )
        assert pipe is not None

    def test_all_fields_flow_to_train_model(self, monkeypatch):
        """Каждое поле TrainingOptions должно дойти до _augment_training_data.

        monkeypatched internal helper — bump if rename.
        """
        import ml_training as mt

        captured: dict = {}

        def _spy_augment(X, y, *, random_state, **kwargs):
            captured.update(kwargs)
            captured["random_state"] = random_state
            return list(X), list(y)

        monkeypatch.setattr(mt, "_augment_training_data", _spy_augment)

        opts = TrainingOptions(
            calib_method="isotonic",
            use_smote=True,
            oversample_strategy="augment_light",
            max_dup_per_sample=7,
            use_hard_negatives=True,
            use_field_dropout=True,
            field_dropout_prob=0.25,
            field_dropout_copies=3,
            use_label_smoothing=True,
            label_smoothing_eps=0.08,
            use_fuzzy_dedup=True,
            fuzzy_dedup_threshold=88,
        )

        X, y = _tiny_dataset()
        train_model(
            X, y, _simple_vectorizer(),
            C=1.0, max_iter=200, balanced=True,
            test_size=0.2, random_state=42,
            options=opts,
        )

        assert captured["use_smote"] is True
        assert captured["oversample_strategy"] == "augment_light"
        assert captured["max_dup_per_sample"] == 7
        assert captured["use_hard_negatives"] is True
        assert captured["use_field_dropout"] is True
        assert captured["field_dropout_prob"] == pytest.approx(0.25)
        assert captured["field_dropout_copies"] == 3
        assert captured["use_label_smoothing"] is True
        assert captured["label_smoothing_eps"] == pytest.approx(0.08)
        assert captured["use_fuzzy_dedup"] is True
        assert captured["fuzzy_dedup_threshold"] == 88

    def test_deprecation_stacklevel_points_to_caller(self):
        """stacklevel должен показывать на тестовый файл, не на ml_training.py."""
        X, y = _tiny_dataset()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            train_model(
                X, y, _simple_vectorizer(),
                C=1.0, max_iter=200, balanced=True,
                test_size=0.2, random_state=42,
                use_smote=False,
            )
        deprec = [w for w in record if issubclass(w.category, DeprecationWarning)]
        assert deprec, "ожидался хотя бы один DeprecationWarning"
        assert deprec[0].filename.endswith("test_training_options.py"), (
            f"stacklevel указывает не на caller: {deprec[0].filename}"
        )
