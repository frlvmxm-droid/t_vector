from ml_vectorizers import Lemmatizer


def test_lemmatizer_falls_back_to_noop_when_morph_init_fails(monkeypatch):
    class _BrokenPymorphy:
        class MorphAnalyzer:
            def __init__(self):
                raise ValueError("inspect compatibility issue")

    monkeypatch.setitem(__import__("sys").modules, "pymorphy2", _BrokenPymorphy)
    lem = Lemmatizer().fit(["Привет мир"])
    assert lem.is_active_ is False
    out = lem.transform(["Тестовый текст"])
    assert out == ["Тестовый текст"]
