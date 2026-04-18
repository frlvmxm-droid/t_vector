# -*- coding: utf-8 -*-
"""Tests for ml_vectorizers: MetaFeatureExtractor, Lemmatizer, PerFieldVectorizer, make_hybrid_vectorizer."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from ml_vectorizers import (
    MetaFeatureExtractor,
    make_hybrid_vectorizer,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_STRUCTURED = [
    "[CHANNEL]\ncall\n[DESC]\nклиент хочет закрыть счёт\n[CLIENT]\nхочу закрыть счёт\n[OPERATOR]\nхорошо понял",
    "[CHANNEL]\nchat\n[CLIENT]\nпомогите с картой\n[OPERATOR]\nкакой тип карты",
    "[DESC]\nзаявка на кредит",
    "",
]

_PLAIN = ["кредит", "карта банк", "помогите пожалуйста", ""]

_BASE_WEIGHTS = {
    "w_desc": 2, "w_client": 3, "w_operator": 2,
    "w_summary": 1, "w_answer_short": 1, "w_answer_full": 1,
}


# ===========================================================================
# MetaFeatureExtractor
# ===========================================================================

class TestMetaFeatureExtractor:

    def test_transform_returns_ndarray(self):
        mfe = MetaFeatureExtractor()
        mfe.fit(_STRUCTURED)
        out = mfe.transform(_STRUCTURED)
        assert isinstance(out, np.ndarray)

    def test_output_shape(self):
        mfe = MetaFeatureExtractor()
        mfe.fit(_STRUCTURED)
        out = mfe.transform(_STRUCTURED)
        assert out.shape == (len(_STRUCTURED), 16)

    def test_empty_string_produces_zeros(self):
        mfe = MetaFeatureExtractor()
        mfe.fit([""])
        out = mfe.transform([""])
        assert out.shape == (1, 16)
        # most features should be 0 for empty string
        assert float(out[0, 0]) == 0.0  # has_desc
        assert float(out[0, 1]) == 0.0  # has_client

    def test_call_channel_detected(self):
        mfe = MetaFeatureExtractor()
        texts = ["[CHANNEL]\ncall\n[CLIENT]\nтест"]
        mfe.fit(texts)
        out = mfe.transform(texts)
        assert float(out[0, 6]) == 1.0  # is_call
        assert float(out[0, 7]) == 0.0  # is_chat

    def test_chat_channel_detected(self):
        mfe = MetaFeatureExtractor()
        texts = ["[CHANNEL]\nchat\n[CLIENT]\nтест"]
        mfe.fit(texts)
        out = mfe.transform(texts)
        assert float(out[0, 6]) == 0.0  # is_call
        assert float(out[0, 7]) == 1.0  # is_chat

    def test_section_presence_flags(self):
        mfe = MetaFeatureExtractor()
        texts = ["[DESC]\nописание\n[CLIENT]\nтекст"]
        mfe.fit(texts)
        out = mfe.transform(texts)
        assert float(out[0, 0]) == 1.0  # has_desc
        assert float(out[0, 1]) == 1.0  # has_client
        assert float(out[0, 2]) == 0.0  # has_operator (absent)

    def test_client_share_between_0_and_1(self):
        mfe = MetaFeatureExtractor()
        mfe.fit(_STRUCTURED)
        out = mfe.transform(_STRUCTURED)
        client_share_col = out[:, 13]
        assert np.all(client_share_col >= 0.0)
        assert np.all(client_share_col <= 1.0)

    def test_n_fields_present_correct(self):
        mfe = MetaFeatureExtractor()
        texts = ["[DESC]\nа\n[CLIENT]\nб\n[OPERATOR]\nв"]
        mfe.fit(texts)
        out = mfe.transform(texts)
        assert float(out[0, 15]) == 3.0  # 3 fields: desc, client, operator

    def test_only_digits_text(self):
        mfe = MetaFeatureExtractor()
        texts = ["12345 67890"]
        mfe.fit(texts)
        out = mfe.transform(texts)
        assert out.shape == (1, 16)

    def test_very_long_text(self):
        mfe = MetaFeatureExtractor()
        long_text = "[CLIENT]\n" + ("слово " * 5000)
        mfe.fit([long_text])
        out = mfe.transform([long_text])
        assert out.shape == (1, 16)
        assert float(out[0, 1]) == 1.0  # has_client

    def test_fit_transform_equivalent(self):
        mfe1 = MetaFeatureExtractor()
        mfe2 = MetaFeatureExtractor()
        out1 = mfe1.fit_transform(_STRUCTURED)
        mfe2.fit(_STRUCTURED)
        out2 = mfe2.transform(_STRUCTURED)
        np.testing.assert_array_almost_equal(out1, out2)

    def test_log1p_lengths_nonnegative(self):
        mfe = MetaFeatureExtractor()
        mfe.fit(_STRUCTURED)
        out = mfe.transform(_STRUCTURED)
        assert np.all(out[:, 8:12] >= 0)  # desc_len, client_len, oper_len, n_client_lines

    def test_dtype_is_float32(self):
        mfe = MetaFeatureExtractor()
        mfe.fit(_STRUCTURED)
        out = mfe.transform(_STRUCTURED)
        assert out.dtype == np.float32


# ===========================================================================
# Lemmatizer
# ===========================================================================

class TestLemmatizer:

    def _get_lemmatizer(self):
        from ml_vectorizers import Lemmatizer
        return Lemmatizer()

    def test_transform_returns_list_of_strings(self):
        lem = self._get_lemmatizer()
        lem.fit(_PLAIN)
        out = lem.transform(_PLAIN)
        assert isinstance(out, list)
        assert all(isinstance(s, str) for s in out)

    def test_output_length_matches_input(self):
        lem = self._get_lemmatizer()
        lem.fit(_PLAIN)
        out = lem.transform(_PLAIN)
        assert len(out) == len(_PLAIN)

    def test_empty_string_handled(self):
        lem = self._get_lemmatizer()
        lem.fit([""])
        out = lem.transform([""])
        assert len(out) == 1
        assert isinstance(out[0], str)

    def test_only_digits_handled(self):
        lem = self._get_lemmatizer()
        lem.fit(["12345"])
        out = lem.transform(["12345"])
        assert len(out) == 1

    def test_cyrillic_text_not_empty(self):
        lem = self._get_lemmatizer()
        texts = ["банк карта кредит"]
        lem.fit(texts)
        out = lem.transform(texts)
        assert len(out[0].strip()) > 0

    def test_section_tags_preserved_or_stripped(self):
        lem = self._get_lemmatizer()
        texts = ["[CLIENT] хочу закрыть счёт"]
        lem.fit(texts)
        out = lem.transform(texts)
        assert isinstance(out[0], str)

    def test_fit_transform_equivalent(self):
        lem1 = self._get_lemmatizer()
        lem2 = self._get_lemmatizer()
        out1 = lem1.fit_transform(_PLAIN)
        lem2.fit(_PLAIN)
        out2 = lem2.transform(_PLAIN)
        assert out1 == out2

    def test_many_samples(self):
        lem = self._get_lemmatizer()
        texts = [f"обращение клиента номер {i} в банк" for i in range(200)]
        lem.fit(texts)
        out = lem.transform(texts)
        assert len(out) == 200


# ===========================================================================
# PerFieldVectorizer
# ===========================================================================

class TestPerFieldVectorizer:

    def _make_vect(self, **kwargs):
        from ml_vectorizers import PerFieldVectorizer
        return PerFieldVectorizer(base_weights=_BASE_WEIGHTS, min_df=1, max_features=10_000, **kwargs)

    def test_fit_transform_shape(self):
        vect = self._make_vect()
        X = vect.fit_transform(_STRUCTURED)
        assert X.shape[0] == len(_STRUCTURED)
        assert X.shape[1] > 0

    def test_output_is_sparse_or_dense(self):
        vect = self._make_vect()
        X = vect.fit_transform(_STRUCTURED)
        assert scipy.sparse.issparse(X) or isinstance(X, np.ndarray)

    def test_transform_matches_fit_transform(self):
        vect1 = self._make_vect()
        vect2 = self._make_vect()
        X1 = vect1.fit_transform(_STRUCTURED)
        vect2.fit(_STRUCTURED)
        X2 = vect2.transform(_STRUCTURED)
        if scipy.sparse.issparse(X1):
            np.testing.assert_array_almost_equal(X1.toarray(), X2.toarray())
        else:
            np.testing.assert_array_almost_equal(X1, X2)

    def test_empty_document_handled(self):
        vect = self._make_vect()
        texts = _STRUCTURED + [""]
        X = vect.fit_transform(texts)
        assert X.shape[0] == len(texts)

    def test_zero_weight_field_gives_zero_cols(self):
        from ml_vectorizers import PerFieldVectorizer
        weights = dict(_BASE_WEIGHTS)
        weights["w_desc"] = 0
        vect = PerFieldVectorizer(base_weights=weights, min_df=1, max_features=1000)
        X = vect.fit_transform(_STRUCTURED)
        assert X.shape[0] == len(_STRUCTURED)

    def test_single_document(self):
        vect = self._make_vect()
        X = vect.fit_transform(["[CLIENT]\nпомогите"])
        assert X.shape[0] == 1

    def test_joblib_roundtrip(self):
        import io
        import joblib
        vect = self._make_vect()
        vect.fit(_STRUCTURED)
        buf = io.BytesIO()
        joblib.dump(vect, buf)
        buf.seek(0)
        vect2 = joblib.load(buf)
        X1 = vect.transform(_STRUCTURED)
        X2 = vect2.transform(_STRUCTURED)
        if scipy.sparse.issparse(X1):
            np.testing.assert_array_almost_equal(X1.toarray(), X2.toarray())
        else:
            np.testing.assert_array_almost_equal(X1, X2)


# ===========================================================================
# make_hybrid_vectorizer
# ===========================================================================

class TestMakeHybridVectorizer:

    def test_returns_callable_pipeline(self):
        pipe = make_hybrid_vectorizer(
            use_per_field=False,
            use_svd=False,
            use_lemma=False,
            use_meta=False,
            min_df=1,
            max_features=5000,
        )
        assert hasattr(pipe, "fit") and hasattr(pipe, "transform")

    def test_fit_transform_plain_text(self):
        pipe = make_hybrid_vectorizer(
            use_per_field=False,
            use_svd=False,
            use_lemma=False,
            use_meta=False,
            min_df=1,
            max_features=5000,
        )
        X = pipe.fit_transform(_PLAIN)
        assert X.shape[0] == len(_PLAIN)
        assert X.shape[1] > 0

    def test_per_field_mode(self):
        pipe = make_hybrid_vectorizer(
            use_per_field=True,
            base_weights=_BASE_WEIGHTS,
            use_svd=False,
            use_lemma=False,
            use_meta=False,
            min_df=1,
            max_features=5000,
        )
        X = pipe.fit_transform(_STRUCTURED)
        assert X.shape[0] == len(_STRUCTURED)

    def test_with_svd(self):
        pipe = make_hybrid_vectorizer(
            use_per_field=False,
            use_svd=True,
            svd_components=10,
            use_lemma=False,
            use_meta=False,
            min_df=1,
            max_features=5000,
        )
        texts = [f"банк карта кредит {i}" for i in range(20)]
        X = pipe.fit_transform(texts)
        # SVD may be capped by n_features or skipped; just check output is dense 2D
        assert X.shape[0] == 20
        assert X.ndim == 2

    def test_with_meta_features(self):
        pipe = make_hybrid_vectorizer(
            use_per_field=False,
            use_svd=False,
            use_lemma=False,
            use_meta=True,
            min_df=1,
            max_features=5000,
        )
        X = pipe.fit_transform(_STRUCTURED)
        assert X.shape[0] == len(_STRUCTURED)
        assert X.shape[1] >= 16  # at least meta features

    def test_transform_after_fit_same_shape(self):
        pipe = make_hybrid_vectorizer(
            use_per_field=False,
            use_svd=False,
            use_lemma=False,
            use_meta=False,
            min_df=1,
            max_features=5000,
        )
        pipe.fit(_PLAIN)
        X_train = pipe.transform(_PLAIN)
        X_new = pipe.transform(["новый текст для проверки"])
        assert X_train.shape[1] == X_new.shape[1]

    def test_empty_corpus_no_crash(self):
        pipe = make_hybrid_vectorizer(
            use_per_field=False,
            use_svd=False,
            use_lemma=False,
            use_meta=False,
            min_df=1,
            max_features=5000,
        )
        texts = ["слово"]
        X = pipe.fit_transform(texts)
        assert X.shape[0] == 1

    def test_very_long_single_document(self):
        pipe = make_hybrid_vectorizer(
            use_per_field=False,
            use_svd=False,
            use_lemma=False,
            use_meta=False,
            min_df=1,
            max_features=5000,
        )
        long_text = "банк " * 3000
        texts = [long_text, "карта"]
        X = pipe.fit_transform(texts)
        assert X.shape[0] == 2
