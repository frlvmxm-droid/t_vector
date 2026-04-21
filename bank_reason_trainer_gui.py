# -*- coding: utf-8 -*-
"""
bank_reason_trainer_gui.py — УСТАРЕВШАЯ точка входа (legacy).

.. deprecated::
    Используйте bootstrap_run.py вместо этого файла.
    Этот модуль оставлен только для обратной совместимости.

Основная точка входа: bootstrap_run.py (используется в .spec для PyInstaller).

Вся логика вынесена в отдельные модули:
  constants.py       — пути, константы, ModelConfig
  text_utils.py      — strip_html, normalize_text, parse_dialog_roles
  feature_builder.py — build_feature_text, choose_row_profile_weights
  excel_utils.py     — read_headers, idx_of, estimate_total_rows, fmt_eta, fmt_speed
  ml_core.py         — make_hybrid_vectorizer, train_model, extract_cluster_keywords, …
  ui_theme.py        — apply_dark_theme
  ui_widgets_tk.py   — Tooltip, ScrollableFrame, ImageBackground
  app.py             — класс App (главное окно)
"""
if __name__ == "__main__":
    import warnings
    warnings.warn(
        "bank_reason_trainer_gui.py is deprecated. Use bootstrap_run.py instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    from app import App
    App().mainloop()
