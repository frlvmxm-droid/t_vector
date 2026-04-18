from app_cluster import ClusterTabMixin


def test_ollama_defaults_set_model_and_clear_key_when_empty_model():
    model, api_key = ClusterTabMixin._llm_provider_ui_defaults("ollama", "", "secret")
    assert model == "qwen3:8b"
    assert api_key == ""


def test_ollama_defaults_keep_existing_model():
    model, api_key = ClusterTabMixin._llm_provider_ui_defaults("ollama", "qwen3:30b", "secret")
    assert model == "qwen3:30b"
    assert api_key == ""


def test_non_ollama_defaults_do_not_mutate_values():
    model, api_key = ClusterTabMixin._llm_provider_ui_defaults("openai", "gpt-4o-mini", "abc")
    assert model == "gpt-4o-mini"
    assert api_key == "abc"
