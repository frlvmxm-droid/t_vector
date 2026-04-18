# -*- coding: utf-8 -*-
"""
Скрипт для предварительного скачивания всех SBERT-моделей в папку sbert_models/.

Запуск:
    python download_sbert_models.py           # скачать все модели
    python download_sbert_models.py tiny      # только rubert-tiny2 (~45 МБ)
    python download_sbert_models.py large     # только большие модели

После скачивания приложение не будет обращаться к интернету при обучении.
"""
from __future__ import annotations
import sys
import pathlib

# ── Папка назначения ─────────────────────────────────────────────────────────
APP_ROOT = pathlib.Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
from config.ml_constants import hf_cache_key  # noqa: E402
SBERT_DIR = APP_ROOT / "sbert_models"
SBERT_DIR.mkdir(exist_ok=True)

# ── Список моделей ────────────────────────────────────────────────────────────
MODELS: dict[str, str] = {
    "cointegrated/rubert-tiny2":
        "ruBERT-tiny2  (~45 МБ)  — быстрый, хорошее качество для русского",
    "ai-forever/sbert_large_nlu_ru":
        "sbert_large   (~800 МБ) — лучшее качество для русского",
    "cointegrated/LaBSE-en-ru":
        "LaBSE-en-ru   (~470 МБ) — многоязычный EN+RU",
}

TINY_MODELS  = ["cointegrated/rubert-tiny2"]
LARGE_MODELS = [m for m in MODELS if m not in TINY_MODELS]

# ── Аргумент командной строки ─────────────────────────────────────────────────
arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
if arg == "tiny":
    to_download = TINY_MODELS
elif arg == "large":
    to_download = LARGE_MODELS
else:
    to_download = list(MODELS.keys())


def _check_installed() -> bool:
    try:
        import huggingface_hub  # noqa
        return True
    except ImportError:
        return False


def _install_deps() -> None:
    import subprocess
    print("▶ Устанавливаю huggingface_hub и sentence-transformers…")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "huggingface-hub", "sentence-transformers",
    ])
    print("✅ Зависимости установлены.\n")


# ── Основной цикл скачивания ──────────────────────────────────────────────────
def main() -> None:
    if not _check_installed():
        _install_deps()

    from huggingface_hub import snapshot_download

    total = len(to_download)
    for i, model_id in enumerate(to_download, 1):
        desc = MODELS.get(model_id, "")
        print(f"\n[{i}/{total}] Скачиваю: {model_id}")
        print(f"         {desc}")
        dest = SBERT_DIR / hf_cache_key(model_id)
        if dest.exists() and any(dest.iterdir()):
            print(f"  ✅ Уже скачана: {dest}")
            continue
        try:
            snapshot_download(
                model_id,
                cache_dir=str(SBERT_DIR),
                ignore_patterns=["*.h5", "*.ot", "flax_model*", "tf_model*",
                                  "rust_model*", "onnx*"],
            )
            print(f"  ✅ Готово → {SBERT_DIR}")
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")

    print(f"\n{'='*60}")
    print(f"Скачивание завершено. Модели сохранены в:")
    print(f"  {SBERT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
