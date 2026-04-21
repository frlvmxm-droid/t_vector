# -*- coding: utf-8 -*-
"""
app.py — класс App (GUI): главное окно приложения.
Импортирует всё из вспомогательных модулей.

Методы, специфичные для каждой вкладки, вынесены в миксины:
  • TrainTabMixin   (app_train.py)   — вкладка «Обучение»
  • ApplyTabMixin   (app_apply.py)   — вкладка «Классификация»
  • ClusterTabMixin (app_cluster.py) — вкладка «Кластеризация»
"""
from __future__ import annotations

import sys
import subprocess

# Подавляем системные диалоги Windows о сбоях DLL (например, несовместимый
# torchvision/_C.pyd после смены версии torch). Без этого Windows показывает
# модальный диалог "Точка входа не найдена" до того, как Python может
# перехватить исключение. Устанавливаем SEM_FAILCRITICALERRORS (0x1) +
# SEM_NOOPENFILEERRORBOX (0x8000) — ошибки превращаются в Python-исключения.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x8001)
    except (ImportError, AttributeError, OSError):
        pass  # ctypes недоступен или API отсутствует — безопасно игнорировать

# pystray — иконка в системном трее (опционально)
# Ловим Exception (не только ImportError): на headless Linux pystray может
# поднять Xlib.error.DisplayNameError или другие не-ImportError ошибки.
try:
    import pystray as _pystray
    from PIL import Image as _PILTrayImg, ImageDraw as _PILTrayDraw
    _HAS_PYSTRAY = True
except (ImportError, OSError, RuntimeError) as _e:
    import logging as _logging_tray
    _logging_tray.getLogger(__name__).warning(
        "pystray недоступен, иконка трея отключена: reason=%s", _e)
    _HAS_PYSTRAY = False

import threading
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import customtkinter as ctk

from constants import (
    APP_ROOT, MODEL_DIR, CLASS_DIR, CLUST_DIR,
)
from config import (
    DEFAULT_COLS,
    SBERT_MODELS_LIST, SBERT_DEFAULT,
    DEFAULT_OTHER_LABEL,
    SETFIT_MODELS_LIST, SETFIT_DEFAULT,
    load_exclusions, save_exclusions,
)
from text_utils import normalize_text, parse_dialog_roles, clean_answer_text
from feature_builder import build_feature_text, choose_row_profile_weights
from excel_utils import idx_of
from ml_core import SBERTVectorizer

import hw_profile as _hw_module
from ui_theme import apply_dark_theme, _best_font, BG, PANEL, PANEL2, FG, MUTED, MUTED2, ENTRY_BG, BORDER, ACCENT, ACCENT2, SELECT, SUCCESS, WARNING, ERROR, HOVER
from ui_widgets_tk import Tooltip, ScrollableFrame, ImageBackground, RoundedButton, PillTabBar, ToggleSwitch

from ui_theme_ctk import apply_theme as _ctk_apply_theme, COLORS as CTK_COLORS, font_label, font_md_bold, font_mono, font_sm
from app_train_view_ctk import build_train_tab
from app_apply_view_ctk import build_apply_tab
from app_cluster_view_ctk import build_cluster_tab

from app_train import TrainTabMixin
from app_apply import ApplyTabMixin
from app_cluster import ClusterTabMixin
from app_deps import DepsTabMixin
from app_logger import get_logger
from model_loader import TrustStore

_log = get_logger(__name__)

_USER_DIR    = Path.home() / ".classification_tool"
SESSION_FILE = _USER_DIR / "last_session.json"
RECENTS_FILE = _USER_DIR / "recents.json"
PRESETS_DIR  = _USER_DIR / "presets"
_MAX_RECENTS = 8


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class App(DepsTabMixin, TrainTabMixin, ApplyTabMixin, ClusterTabMixin, ctk.CTk):
    # ------------------------------------------------------------------ init
    def __init__(self):
        _ctk_apply_theme("paper-dark")   # должно быть до super().__init__()
        super().__init__()
        self.title("Hearsy · Classifier")

        # ── Адаптивный размер окна: 92% экрана, но не меньше 1200×760 ───────
        self.update_idletasks()
        _sw = self.winfo_screenwidth()
        _sh = self.winfo_screenheight()
        _W = min(1500, max(1200, int(_sw * 0.92)))
        _H = min(980,  max(760,  int(_sh * 0.92)))
        self.minsize(min(1200, _W), min(760, _H))
        _wx, _wy = max(0, (_sw - _W) // 2), max(0, (_sh - _H) // 2)
        self.geometry(f"{_W}x{_H}+{_wx}+{_wy}")

        self.configure(fg_color=CTK_COLORS["bg"])

        # Определяем характеристики ПК ПЕРВЫМИ — их значения используются
        # как дефолты для tkvar ниже.
        self._hw = _hw_module.detect()

        self.base_font = 10
        self.zoom = tk.DoubleVar(value=1.0)
        self._apply_window_icon()

        self._init_file_vars()
        self._init_feature_vars()
        self._init_model_vars()
        self._init_cluster_vars()

        self._processing = False
        self._proc_lock = threading.Lock()   # защита от race condition при двойном клике
        self._cancel_event = threading.Event()  # устанавливается кнопкой «Стоп»
        self._headers_cache: List[str] = []

        self._user_exclusions: dict = load_exclusions()
        # backward-compat: _custom_stop_words — stop_words раздел (используется в snap)
        self._custom_stop_words: List[str] = self._user_exclusions.get("stop_words", [])

        self._current_tab: str = "train"

        self._tray_icon: Any = None

        self._build_ui()
        self._bind_guardrails()
        self._apply_gpu_optimal_params()
        self._apply_zoom()
        # Выводим GPU-статус в лог после инициализации UI
        self.after(200, self._log_gpu_startup)
        # Запускаем иконку в трее (если pystray доступен)
        self.after(500, self._start_tray)
        # Восстанавливаем прошлую сессию (тихо, без диалогов)
        self.after(100, self._restore_session)
        # Горячие клавиши
        self.bind_all("<F5>",            lambda e: self._run_active_tab())
        self.bind_all("<Control-Return>", lambda e: self._run_active_tab())
        self.bind_all("<Escape>",        lambda e: self._request_cancel())
        self.bind_all("<Control-Key-1>", lambda e: self._hotkey_switch_tab(0))
        self.bind_all("<Control-Key-2>", lambda e: self._hotkey_switch_tab(1))
        self.bind_all("<Control-Key-3>", lambda e: self._hotkey_switch_tab(2))
        self.bind_all("<Control-h>",     lambda e: self._open_history_dialog())
        self.bind_all("<Control-comma>", lambda e: self._open_settings_dialog())
        # Обновляем badges при изменении apply_file
        self.apply_file.trace_add("write", lambda *_: self.after_idle(self._update_badges))

    def _apply_window_icon(self) -> None:
        """Подключает иконки окна/панели задач из файлов проекта (если есть)."""
        base = Path(__file__).resolve().parent
        icon_ico_candidates = [
            base / "ui" / "app_icon.ico",
            base / "ui" / "icon.ico",
            base / "app_icon.ico",
            base / "icon.ico",
        ]
        icon_png_candidates = [
            base / "ui" / "app_icon.png",
            base / "ui" / "icon.png",
            base / "app_icon.png",
            base / "icon.png",
        ]
        try:
            for p in icon_ico_candidates:
                if p.exists():
                    self.iconbitmap(str(p))
                    break
        except Exception as _e:
            _log.debug("iconbitmap load failed: %s", _e)
        try:
            for p in icon_png_candidates:
                if p.exists():
                    self._app_icon_tk = tk.PhotoImage(file=str(p))
                    self.iconphoto(True, self._app_icon_tk)
                    break
        except Exception as _e:
            _log.debug("iconphoto load failed: %s", _e)

    # ------------------------------------------------------------ GPU device list

    @property
    def gpu_device_values(self) -> List[str]:
        """Список допустимых значений для device combo-боксов.

        1 GPU:  ['auto', 'cpu', 'cuda', 'cuda:0']
        2 GPU:  ['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1']
        """
        return ["auto", "cpu", "cuda"] + self._hw.gpu_devices

    # ---------------------------------------------------------------- var init

    def _init_file_vars(self) -> None:
        """Инициализирует переменные файлов и имён колонок."""
        self.train_files: List[str] = []
        self.base_model_file = tk.StringVar(value="")
        self.train_mode = tk.StringVar(value="new")

        self.model_file = tk.StringVar(value="")
        self.apply_file = tk.StringVar(value="")
        # Хранилище доверенных путей моделей для текущей сессии
        self._trust_store: TrustStore = TrustStore()
        self.cluster_file = tk.StringVar(value="")  # kept for compatibility
        self.cluster_files: List[str] = []

        self.desc_col = tk.StringVar(value=DEFAULT_COLS["desc"][0])
        self.call_col = tk.StringVar(value=DEFAULT_COLS["call"][0])
        self.chat_col = tk.StringVar(value=DEFAULT_COLS["chat"][0])
        self.summary_col = tk.StringVar(value=DEFAULT_COLS["summary"][0])
        self.ans_short_col = tk.StringVar(value=DEFAULT_COLS["ans_short"][0])
        self.ans_full_col = tk.StringVar(value=DEFAULT_COLS["ans_full"][0])
        self.label_col = tk.StringVar(value=DEFAULT_COLS["label"][0])

    def _init_feature_vars(self) -> None:
        """Инициализирует флаги предобработки и веса секций текста."""
        self.use_summary = tk.BooleanVar(value=True)
        self.ignore_chatbot = tk.BooleanVar(value=True)
        self.auto_profile = tk.StringVar(value="smart")  # off|smart|strict
        self.use_stop_words = tk.BooleanVar(value=True)
        self.use_noise_tokens = tk.BooleanVar(value=True)
        self.use_noise_phrases = tk.BooleanVar(value=True)

        self.w_desc = tk.IntVar(value=2)
        self.w_client = tk.IntVar(value=3)
        self.w_operator = tk.IntVar(value=1)
        self.w_summary = tk.IntVar(value=2)
        self.w_ans_short = tk.IntVar(value=1)
        self.w_ans_full = tk.IntVar(value=1)

    def _init_model_vars(self) -> None:
        """Инициализирует параметры TF-IDF, SVC, SBERT, SetFit и применения модели."""
        # TF-IDF / SVC
        self.char_ng_min = tk.IntVar(value=2)
        self.char_ng_max = tk.IntVar(value=9)
        self.word_ng_min = tk.IntVar(value=1)
        self.word_ng_max = tk.IntVar(value=3)
        self.min_df = tk.IntVar(value=3)
        self.max_features = tk.IntVar(value=self._hw.max_features)
        self.sublinear_tf = tk.BooleanVar(value=True)
        self.C = tk.DoubleVar(value=3.0)
        self.max_iter = tk.IntVar(value=2000)

        # GPU master switch — переключает все GPU-способные операции на CUDA
        self.use_gpu_all = tk.BooleanVar(value=SBERTVectorizer.is_cuda_available())

        # SBERT
        self.use_sbert = tk.BooleanVar(value=False)
        self.sbert_model = tk.StringVar(value=SBERT_DEFAULT)
        self.sbert_device = tk.StringVar(value="auto")   # "auto" | "cpu" | "cuda"
        self.sbert_batch = tk.IntVar(value=self._hw.sbert_batch)
        self.use_per_field = tk.BooleanVar(value=True)
        self.use_svd = tk.BooleanVar(value=True)
        self.svd_components = tk.IntVar(value=200)
        self.use_lemma = tk.BooleanVar(value=True)
        self.use_meta = tk.BooleanVar(value=False)
        self.use_sbert_hybrid = tk.BooleanVar(value=False)
        self.use_smote = tk.BooleanVar(value=True)
        self.drop_conflicts = tk.BooleanVar(value=True)
        self.use_llm_augment      = tk.BooleanVar(value=False)
        self.augment_min_samples  = tk.IntVar(value=30)
        self.augment_n_paraphrases = tk.IntVar(value=3)
        self.detect_near_dups     = tk.BooleanVar(value=False)
        self.near_dup_threshold   = tk.DoubleVar(value=0.92)
        self.use_hard_negatives   = tk.BooleanVar(value=False)
        self.use_field_dropout    = tk.BooleanVar(value=False)
        self.field_dropout_prob   = tk.DoubleVar(value=0.15)
        self.field_dropout_copies = tk.IntVar(value=2)
        # Нормализация сущностей
        self.use_entity_norm      = tk.BooleanVar(value=False)
        # Детекция ошибочных меток
        self.detect_mislabeled    = tk.BooleanVar(value=False)
        self.mislabeled_threshold = tk.DoubleVar(value=0.30)
        # Псевдо-разметка
        self.use_pseudo_label        = tk.BooleanVar(value=False)
        self.pseudo_label_file       = tk.StringVar(value="")
        self.pseudo_label_threshold  = tk.DoubleVar(value=0.92)
        # Иерархическая классификация
        self.use_hierarchical     = tk.BooleanVar(value=False)
        # Якорные тексты классов
        self.use_anchor_texts     = tk.BooleanVar(value=False)
        self.anchor_copies        = tk.IntVar(value=3)
        # Confident Learning
        self.use_confident_learning         = tk.BooleanVar(value=False)
        self.confident_learning_threshold   = tk.DoubleVar(value=1.0)
        # K-fold ансамбль
        self.use_kfold_ensemble   = tk.BooleanVar(value=False)
        self.kfold_k              = tk.IntVar(value=5)
        # Optuna автоподбор гиперпараметров
        self.use_optuna           = tk.BooleanVar(value=False)
        self.n_optuna_trials      = tk.IntVar(value=30)
        self.calib_method = tk.StringVar(value="sigmoid")
        self.class_weight_balanced = tk.BooleanVar(value=True)
        self.test_size = tk.DoubleVar(value=0.2)

        # SetFit нейросетевой классификатор
        self.use_setfit = tk.BooleanVar(value=False)
        self.setfit_model = tk.StringVar(value=SETFIT_DEFAULT)
        self.setfit_epochs = tk.IntVar(value=3)
        self.setfit_num_iterations = tk.IntVar(value=50)
        self.setfit_batch = tk.IntVar(value=self._hw.setfit_batch)
        self.setfit_fp16 = tk.BooleanVar(value=True)

        # Единый переключатель режима обучения (виджет в CTK-view)
        self.train_vec_mode = tk.StringVar(value="tfidf")

        def _sync_train_mode(*_):
            m = self.train_vec_mode.get()
            self.use_sbert.set(m == "sbert")
            self.use_sbert_hybrid.set(m == "hybrid")
            self.use_setfit.set(m == "setfit")

        self.train_vec_mode.trace_add("write", _sync_train_mode)

        # Ensemble mode (ансамбль нескольких моделей при классификации)
        self.use_ensemble = tk.BooleanVar(value=False)
        self.ensemble_model2 = tk.StringVar(value="")
        self.ensemble_w1 = tk.DoubleVar(value=0.5)  # вес модели 1 (вес модели 2 = 1 - w1)

        # Apply outputs
        self.pred_col = tk.StringVar(value="pred_marker1")
        self.review_threshold = tk.DoubleVar(value=0.60)
        self._rec_threshold_75 = tk.DoubleVar(value=0.0)   # рек. порог из обучения (75-й перцентиль)
        self.use_other_label = tk.BooleanVar(value=False)
        self.other_label_threshold = tk.DoubleVar(value=0.50)
        self.other_label_text = tk.StringVar(value=DEFAULT_OTHER_LABEL)
        # Использовать per-class пороги (из обученной модели) для «Другое»;
        # глобальный other_label_threshold становится fallback.
        self.use_per_class_other_threshold = tk.BooleanVar(value=False)
        # Кастомные per-class пороги (редактируемые без переобучения).
        # Ключ — метка класса, значение — DoubleVar с порогом.
        self._custom_class_thresholds: dict = {}
        # Признак "использовать кастомные пороги" (override модельных значений).
        self.use_custom_class_thresholds = tk.BooleanVar(value=False)
        # Стратегия оверсэмплинга: cap | duplicate | augment_light | nn_mix.
        # nn_mix — SMOTE-подобное смешение ближайших соседей по TF-IDF.
        self.oversample_strategy = tk.StringVar(value="augment_light")
        # LLM-ре-ранк для неуверенных предсказаний (confidence ∈ [low, high]).
        self.use_llm_rerank = tk.BooleanVar(value=False)
        self.llm_rerank_low = tk.DoubleVar(value=0.50)
        self.llm_rerank_high = tk.DoubleVar(value=0.70)
        self.llm_rerank_top_k = tk.IntVar(value=3)
        # Label smoothing для обучения: мягкие метки в CalibratedClassifierCV.
        self.use_label_smoothing = tk.BooleanVar(value=False)
        self.label_smoothing_eps = tk.DoubleVar(value=0.05)
        # POS-теги в лемматизаторе (pymorphy2) — добавляет признаки типа «снять_VERB»
        # вместо «снять» (различает части речи; полезно в русском с омонимами).
        self.use_pos_tags = tk.BooleanVar(value=False)
        self.use_ambiguity_detector = tk.BooleanVar(value=False)
        self.ambiguity_epsilon = tk.DoubleVar(value=0.07)
        self._rec_thr_label = tk.StringVar(value="(сначала обучите модель)")

        # Progress bars — обучение, применение, кластеризация
        self.train_progress = tk.DoubleVar(value=0.0)
        self.train_status   = tk.StringVar(value="")
        self.train_pct      = tk.StringVar(value="")
        self.train_phase    = tk.StringVar(value="")
        self.train_speed    = tk.StringVar(value="")
        self.train_eta      = tk.StringVar(value="")

        self.apply_progress = tk.DoubleVar(value=0.0)
        self.apply_status   = tk.StringVar(value="")
        self.apply_pct      = tk.StringVar(value="")
        self.apply_phase    = tk.StringVar(value="")
        self.apply_speed    = tk.StringVar(value="")
        self.apply_eta      = tk.StringVar(value="")

        self.cluster_progress = tk.DoubleVar(value=0.0)
        self.cluster_status   = tk.StringVar(value="")
        self.cluster_pct      = tk.StringVar(value="")
        self.cluster_phase    = tk.StringVar(value="")
        self.cluster_speed    = tk.StringVar(value="")
        self.cluster_eta      = tk.StringVar(value="")

        self.last_apply_summary = tk.StringVar(value="")
        self.last_cluster_summary = tk.StringVar(value="")

    def _init_cluster_vars(self) -> None:
        """Инициализирует параметры кластеризации."""
        self.k_clusters = tk.IntVar(value=20)
        self.n_init_cluster = tk.IntVar(value=15)
        self.use_elbow = tk.BooleanVar(value=True)
        self.cluster_id_col = tk.StringVar(value="cluster_id")
        self.cluster_kw_col = tk.StringVar(value="cluster_keywords")
        self.use_sbert_cluster = tk.BooleanVar(value=False)  # kept for snap compat
        self.ignore_chatbot_cluster = tk.BooleanVar(value=True)
        self.cluster_role_mode = tk.StringVar(value="client")  # all | client | operator
        self.use_combo_cluster = tk.BooleanVar(value=False)  # kept for snap compat
        self.combo_svd_dim = tk.IntVar(value=200)
        self.combo_alpha = tk.DoubleVar(value=0.5)
        self.k_score_method = tk.StringVar(value="calinski")  # elbow | silhouette | calinski
        # cosine distance устраняет влияние длины документа для TF-IDF и SBERT
        self.use_cosine_cluster = tk.BooleanVar(value=True)
        self.use_umap = tk.BooleanVar(value=False)
        self.umap_n_components = tk.IntVar(value=15)
        self.umap_n_neighbors = tk.IntVar(value=15)
        self.umap_min_dist = tk.DoubleVar(value=0.1)
        self.umap_metric = tk.StringVar(value="cosine")
        self.use_hdbscan = tk.BooleanVar(value=False)  # kept for snap compat
        # 0 = авто: max(5, √N). Для выбранного N даёт в 3–10 раз более разумное
        # значение, чем статичные 10 для любых объёмов.
        self.hdbscan_min_cluster_size = tk.IntVar(value=0)
        # HDBSCAN min_samples: управляет «консервативностью» — чем больше, тем
        # меньше кластеров и больше шума. None = равно min_cluster_size (sklearn default).
        self.hdbscan_min_samples = tk.IntVar(value=0)   # 0 = auto (= min_cluster_size)
        # cluster_selection_epsilon: объединяет кластеры ближе epsilon по дереву.
        # Полезно чтобы уменьшить фрагментацию при банковских диалогах.
        self.hdbscan_eps = tk.DoubleVar(value=0.0)      # 0.0 = не применять

        # Режимы: векторизация и алгоритм
        self.cluster_vec_mode = tk.StringVar(value="tfidf")   # tfidf | sbert | combo | ensemble
        self.cluster_algo = tk.StringVar(value="kmeans")       # kmeans | hdbscan | lda | hierarchical | bertopic
        self.sbert_model2 = tk.StringVar(value=SBERT_DEFAULT)
        self.ensemble_vec2 = tk.StringVar(value="tfidf")       # tfidf | sbert | hybrid

        # LDA (Latent Dirichlet Allocation)
        self.lda_n_topics = tk.IntVar(value=15)
        self.lda_max_iter = tk.IntVar(value=50)
        self.lda_topics_col = tk.StringVar(value="lda_topics")

        # Иерархическая кластеризация (2 уровня)
        self.hier_k_top = tk.IntVar(value=8)
        self.hier_k_sub = tk.IntVar(value=5)
        self.hier_min_sub = tk.IntVar(value=50)
        self.hier_l1_col = tk.StringVar(value="cluster_l1")

        # BERTopic-like (SBERT + UMAP + HDBSCAN + c-TF-IDF)
        self.bertopic_min_topic_size = tk.IntVar(value=10)
        self.bertopic_nr_topics = tk.StringVar(value="auto")

        # Предобработка текста
        self.use_lemma_cluster = tk.BooleanVar(value=True)
        self.normalize_numbers = tk.BooleanVar(value=True)

        # SVD для TF-IDF
        self.use_tfidf_svd = tk.BooleanVar(value=True)
        self.tfidf_svd_dim = tk.IntVar(value=100)

        self.use_ctfidf_keywords = tk.BooleanVar(value=True)
        self.cluster_random_seed = tk.IntVar(value=42)
        self.use_anchors = tk.BooleanVar(value=False)

        # T5 summarization
        self.use_t5_summary = tk.BooleanVar(value=False)
        self.t5_summary_col = tk.StringVar(value="t5_summary")
        self.t5_model_name = tk.StringVar(value="UrukHan/t5-russian-summarization")
        self.t5_max_input = tk.IntVar(value=512)
        self.t5_max_output = tk.IntVar(value=128)
        self.t5_batch_size = tk.IntVar(value=self._hw.t5_batch)

        # LLM-нейминг кластеров
        self.use_llm_naming = tk.BooleanVar(value=False)
        self.llm_api_key = tk.StringVar(value="")
        self.llm_provider = tk.StringVar(value="anthropic")   # anthropic | openai
        self.llm_model = tk.StringVar(value="claude-sonnet-4-6")
        self.llm_name_col = tk.StringVar(value="cluster_name")
        self.use_llm_reason_summary = tk.BooleanVar(value=False)
        self.llm_reason_col = tk.StringVar(value="cluster_reason")
        self.use_rule_reason_summary = tk.BooleanVar(value=True)
        self.merge_similar_clusters = tk.BooleanVar(value=False)
        self.merge_threshold        = tk.DoubleVar(value=0.85)
        self.n_repr_examples        = tk.IntVar(value=5)

        # PCA перед UMAP
        self.use_pca_before_umap = tk.BooleanVar(value=False)
        self.pca_n_components = tk.IntVar(value=50)

        # FASTopic
        self.fastopic_n_top_words = tk.IntVar(value=15)
        self.fastopic_theta = tk.DoubleVar(value=0.15)

        # Аннотация качества кластеров
        self.show_cluster_quality = tk.BooleanVar(value=True)
        self.cluster_quality_col = tk.StringVar(value="cluster_quality")

        # Второй KMeans по выбросам HDBSCAN
        self.recluster_noise = tk.BooleanVar(value=True)
        self.noise_k_clusters = tk.IntVar(value=5)
        self.noise_cluster_col = tk.StringVar(value="noise_cluster_id")

        self.cluster_min_df = tk.IntVar(value=0)

        # Семантическая дедупликация
        self.use_dedup = tk.BooleanVar(value=False)
        self.dedup_threshold = tk.DoubleVar(value=0.95)

        self.use_cluster_viz = tk.BooleanVar(value=False)

        # Инкрементальная кластеризация
        self.save_cluster_model = tk.BooleanVar(value=False)
        self.cluster_model_path = tk.StringVar(value="")
        self.use_saved_cluster_model = tk.BooleanVar(value=False)
        self.diagnostic_mode = tk.BooleanVar(value=False)

        # Потоковый / онлайн-режим (MiniBatchKMeans partial_fit)
        self.use_streaming_cluster = tk.BooleanVar(value=False)
        self.streaming_chunk_size = tk.IntVar(value=5000)

        # ClusterLLM feedback loop
        self.use_llm_feedback = tk.BooleanVar(value=False)
        self.llm_feedback_col = tk.StringVar(value="llm_feedback")

    # ------------------------------------------------------------------ utils
    def _open_directory(self, path: Path) -> None:
        """Открывает папку в проводнике ОС (Windows / macOS / Linux)."""
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as _e:
            _log.debug("open_dir(%s): %s", path, _e)
            messagebox.showwarning(
                "Открытие папки",
                f"Не удалось открыть папку:\n{path}\n\n{_e}",
            )

    # ------------------------------------------------------------------ zoom
    def _apply_zoom(self):
        z = float(self.zoom.get())
        z = max(0.8, min(1.4, z))
        try:
            ctk.set_widget_scaling(z)
        except Exception as _e:
            _log.debug("ctk set_widget_scaling: %s", _e)

    def _zoom_step(self, delta: float):
        v = float(self.zoom.get())
        v = max(0.85, min(1.25, v + delta))
        self.zoom.set(v)
        self._apply_zoom()

    # ------------------------------------------------------------------ help
    def set_help(self, title: str, body: str):
        self.help_title.set(title)
        self.help_box.configure(state="normal")
        self.help_box.delete("1.0", "end")
        self.help_box.insert("1.0", body.strip())
        self.help_box.configure(state="disabled")

    def set_warnings(self, body: str):
        self.warn_box.configure(state="normal")
        self.warn_box.delete("1.0", "end")
        self.warn_box.insert("1.0", body.strip() if body.strip() else "OK ✅")
        self.warn_box.configure(state="disabled")

    def attach_help(self, widget, title: str, body: str, tooltip: str = ""):
        def _on(_e=None):
            self.set_help(title, body)
        widget.bind("<FocusIn>", _on, add="+")
        widget.bind("<Button-1>", _on, add="+")
        if tooltip:
            Tooltip(widget, tooltip)

    # ------------------------------------------------------------------ build UI
    def _build_ui(self):
        # ── CTK sidebar + content layout ─────────────────────────────────────
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._apply_dark_ttk_style()
        self._build_sidebar()
        self._build_main()

        # Stub-виджеты для совместимости с методами set_help/set_warnings / unified_log.
        # Они существуют, но не отображаются.
        self.help_title = tk.StringVar(value="—")
        self.help_box   = tk.Text()   # orphan widget — not packed
        self.warn_box   = tk.Text()
        self.unified_log = tk.Text()

        # Stub-объекты для совместимости с run_training / run_apply / run_cluster,
        # которые вызывают self._right_tabs.select(self._log_tab_indices[...]).
        class _NoOpTabs:
            def select(self, _idx): pass
            def index(self, _tab): return 0
        self._right_tabs = _NoOpTabs()
        self._log_tab_indices = {"train": 0, "apply": 0, "cluster": 0}

        # Stub-фреймы для _restore_session (устаревшие, после миграции)
        self.sf_train = type("_SF", (), {"inner": tk.Frame()})()
        self.sf_apply = type("_SF", (), {"inner": tk.Frame()})()
        self.sf_cluster = type("_SF", (), {"inner": tk.Frame()})()
        self.sf_deps = type("_SF", (), {"inner": tk.Frame()})()
        self.tab_train = self.sf_train.inner
        self.tab_apply = self.sf_apply.inner
        self.tab_cluster = self.sf_cluster.inner
        self.tab_deps = self.sf_deps.inner

    def _apply_dark_ttk_style(self) -> None:
        """Применяет тёмную тему ко всем ttk-виджетам (Treeview, Combobox, Scrollbar)."""
        BG     = CTK_COLORS["bg"]
        PANEL  = CTK_COLORS["panel"]
        PANEL2 = CTK_COLORS["panel2"]
        FG     = CTK_COLORS["fg"]
        MUTED  = CTK_COLORS["muted"]
        ENTRY  = CTK_COLORS["entry"]
        SELECT = CTK_COLORS["select"]
        BORDER = CTK_COLORS["border"]
        HOVER  = CTK_COLORS["hover"]
        s = ttk.Style()
        try:
            s.theme_use("default")
        except Exception:
            pass
        s.configure(".", background=BG, foreground=FG, fieldbackground=ENTRY,
                    selectbackground=SELECT, selectforeground=FG, borderwidth=0)
        # Treeview
        s.configure("Treeview", background=PANEL, foreground=FG,
                    fieldbackground=PANEL, rowheight=26, borderwidth=0, relief="flat")
        s.configure("Treeview.Heading", background=PANEL2, foreground=MUTED,
                    relief="flat", borderwidth=0)
        s.map("Treeview",
              background=[("selected", SELECT)],
              foreground=[("selected", FG)])
        s.map("Treeview.Heading",
              background=[("active", PANEL2)])
        # Scrollbar — trough matches panel2 so the track area is invisible
        for orient in ("Vertical", "Horizontal"):
            s.configure(f"{orient}.TScrollbar",
                        background=PANEL2, troughcolor=PANEL2,
                        darkcolor=PANEL2, lightcolor=PANEL2,
                        bordercolor=PANEL2, arrowcolor=MUTED,
                        borderwidth=0, relief="flat", arrowsize=12)
            s.map(f"{orient}.TScrollbar",
                  background=[("active", MUTED), ("disabled", PANEL2)])
        # Combobox
        s.configure("TCombobox", fieldbackground=ENTRY, background=PANEL,
                    foreground=FG, selectbackground=SELECT, selectforeground=FG,
                    bordercolor=BORDER, insertcolor=FG, relief="flat")
        s.map("TCombobox",
              fieldbackground=[("readonly", ENTRY), ("disabled", BG)],
              foreground=[("disabled", MUTED)],
              background=[("readonly", PANEL), ("active", HOVER)])
        # Frame / Label
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=FG)
        # Combobox dropdown listbox
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", SELECT)
        self.option_add("*TCombobox*Listbox.selectForeground", FG)

    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self, fg_color=CTK_COLORS["panel"], width=220, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsw")
        sb.grid_propagate(False)

        # ── Брендинг ─────────────────────────────────────────────────
        brand = ctk.CTkFrame(sb, fg_color="transparent")
        brand.pack(fill="x", padx=16, pady=(18, 14))
        ctk.CTkLabel(
            brand, text="Hs", width=36, height=36,
            font=font_md_bold(), fg_color=CTK_COLORS["accent"],
            text_color="#ffffff", corner_radius=8,
        ).pack(side="left", padx=(0, 10))
        _brand_txt = ctk.CTkFrame(brand, fg_color="transparent")
        _brand_txt.pack(side="left")
        ctk.CTkLabel(_brand_txt, text="Hearsy", font=font_md_bold(),
                     text_color=CTK_COLORS["fg"]).pack(anchor="w")
        ctk.CTkLabel(_brand_txt, text="CLASSIFIER \u00b7 RU", font=font_label(),
                     text_color=CTK_COLORS["muted"]).pack(anchor="w")

        ctk.CTkFrame(sb, height=1, fg_color=CTK_COLORS["border2"]).pack(fill="x")

        # ── WORKFLOW ─────────────────────────────────────────────────
        ctk.CTkLabel(sb, text="WORKFLOW", font=font_label(),
                     text_color=CTK_COLORS["muted"], anchor="w").pack(
            fill="x", padx=20, pady=(14, 4))

        self._nav_buttons: dict = {}
        self._nav_badges: dict = {}
        self._nav_indicators: dict = {}

        _WORKFLOW = [
            ("train",   "Обучение",       "\u29c9"),
            ("apply",   "Классификация",  "\u25a3"),
            ("cluster", "Кластеризация",  "\u2b21"),
            ("log",     "Логи",           "\u2261"),
        ]
        for key, label, icon in _WORKFLOW:
            is_active = (key == "train")
            _row = ctk.CTkFrame(sb, fg_color="transparent", height=40)
            _row.pack(fill="x")
            _row.pack_propagate(False)

            _ind = ctk.CTkFrame(_row, width=3, corner_radius=0,
                                fg_color=CTK_COLORS["accent"] if is_active else "transparent")
            _ind.pack(side="left", fill="y")
            self._nav_indicators[key] = _ind

            b = ctk.CTkButton(
                _row, text=f"  {icon}  {label}", anchor="w", height=40,
                fg_color="transparent", hover_color=CTK_COLORS["hover"],
                font=font_sm(),
                text_color=CTK_COLORS["fg"] if is_active else CTK_COLORS["muted"],
                command=lambda k=key: self._switch_tab(k),
            )
            b.pack(side="left", fill="both", expand=True)
            self._nav_buttons[key] = b

            if key in ("apply", "cluster"):
                _badge = ctk.CTkLabel(
                    _row, text="0",
                    fg_color=CTK_COLORS["accent"], text_color="#ffffff",
                    corner_radius=9, width=22, height=18, font=font_label(),
                )
                self._nav_badges[key] = _badge

        ctk.CTkFrame(sb, height=1, fg_color=CTK_COLORS["border2"]).pack(
            fill="x", pady=(6, 0))

        # ── КОНТЕКСТ ─────────────────────────────────────────────────
        ctk.CTkLabel(sb, text="КОНТЕКСТ", font=font_label(),
                     text_color=CTK_COLORS["muted"], anchor="w").pack(
            fill="x", padx=20, pady=(12, 4))

        def _open_history():
            from app_dialogs_ctk import HistoryDialog
            HistoryDialog(self).show()

        def _open_artifacts():
            from app_dialogs_ctk import ArtifactsDialog
            ArtifactsDialog(self).show()

        def _open_settings():
            from app_dialogs_ctk import SettingsDialog
            SettingsDialog(self).show()

        for label, cmd, icon in [
            ("Артефакты моделей", _open_artifacts, "\u25a3"),
            ("Настройки",         _open_settings,  "\u2699"),
        ]:
            _crow = ctk.CTkFrame(sb, fg_color="transparent", height=40)
            _crow.pack(fill="x")
            _crow.pack_propagate(False)
            ctk.CTkFrame(_crow, width=3, corner_radius=0,
                         fg_color="transparent").pack(side="left", fill="y")
            ctk.CTkButton(
                _crow, text=f"  {icon}  {label}", anchor="w", height=40,
                fg_color="transparent", hover_color=CTK_COLORS["hover"],
                text_color=CTK_COLORS["muted"], font=font_sm(),
                command=cmd,
            ).pack(side="left", fill="both", expand=True)

        # ── Версия + HW-панель (bottom) ────────────────────────────────
        ctk.CTkFrame(sb, height=1, fg_color=CTK_COLORS["border2"]).pack(
            fill="x", side="bottom")

        try:
            from config.user_config import APP_VERSION
        except Exception:
            APP_VERSION = "v3.4"
        ctk.CTkLabel(sb, text=APP_VERSION, font=font_label(),
                     text_color=CTK_COLORS["muted"], anchor="w").pack(
            fill="x", padx=16, pady=(4, 8), side="bottom")

        hw_card = ctk.CTkFrame(sb, fg_color=CTK_COLORS["panel2"],
                               corner_radius=0, border_width=0)
        hw_card.pack(fill="x", side="bottom")

        _hw_rows = {}
        for _hk, _ht in [("cpu", "CPU  \u2014"), ("ram", "RAM  \u2014"),
                          ("gpu", "GPU  \u2014"), ("torch", "torch  \u2014")]:
            _lbl = ctk.CTkLabel(hw_card, text=_ht, font=font_mono(),
                                text_color=CTK_COLORS["muted"], anchor="w")
            _lbl.pack(fill="x", padx=14, pady=1)
            _hw_rows[_hk] = _lbl
        ctk.CTkLabel(hw_card, text="", height=6).pack()

        def _hw_poll():
            try:
                hw = self._hw
                try:
                    import psutil
                    _pct = int(psutil.cpu_percent(interval=None))
                    _hw_rows["cpu"].configure(text=f"CPU  {hw.cpu_cores} cores \u00b7 {_pct}%")
                    _vm = psutil.virtual_memory()
                    _used = _vm.used / 1e9
                    _total = _vm.total / 1e9
                    _hw_rows["ram"].configure(text=f"RAM  {_used:.1f} / {_total:.0f} \u0413\u0411")
                except Exception:
                    _hw_rows["cpu"].configure(text=f"CPU  {hw.cpu_cores} cores")
                    _hw_rows["ram"].configure(text=f"RAM  {hw.ram_gb:.0f} \u0413\u0411")
                if hw.gpu_name:
                    _vram = f" \u00b7 {hw.gpu_vram_gb:.1f} \u0413\u0411" if hw.gpu_vram_gb else ""
                    _gn = hw.gpu_name.replace("NVIDIA ", "").replace("GeForce ", "")[:20]
                    _hw_rows["gpu"].configure(text=f"GPU  {_gn}{_vram}")
                else:
                    _hw_rows["gpu"].configure(text="GPU  \u2014")
                try:
                    import torch
                    _hw_rows["torch"].configure(text=f"torch  {torch.__version__}")
                except Exception:
                    _hw_rows["torch"].configure(text="torch  \u043d/\u0434")
            except Exception:
                pass
            self.after(3000, _hw_poll)

        _hw_poll()

    def _build_main(self) -> None:
        self.main = ctk.CTkFrame(self, fg_color=CTK_COLORS["bg"], corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.main, fg_color=CTK_COLORS["bg"], height=80,
                               corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        # Left: title + subtitle
        _left_hdr = ctk.CTkFrame(header, fg_color="transparent")
        _left_hdr.pack(side="left", padx=28, pady=12, anchor="w", fill="y")
        self._title = ctk.CTkLabel(
            _left_hdr, text="Обучение модели",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=CTK_COLORS["fg"],
        )
        self._title.pack(anchor="w")
        self._subtitle = ctk.CTkLabel(
            _left_hdr, text="",
            font=font_sm(), text_color=CTK_COLORS["muted"],
        )
        self._subtitle.pack(anchor="w")

        # Right: История + Tweaks
        _right_hdr = ctk.CTkFrame(header, fg_color="transparent")
        _right_hdr.pack(side="right", padx=20, anchor="e", fill="y")
        ctk.CTkButton(
            _right_hdr, text="\u25d7  История", width=120, height=30,
            fg_color="transparent", border_width=1,
            border_color=CTK_COLORS["border"],
            text_color=CTK_COLORS["muted"], font=font_sm(),
            command=lambda: self._open_history_dialog(),
        ).pack(side="left", padx=(0, 8), pady=24)
        ctk.CTkButton(
            _right_hdr, text="\u2261  Tweaks", width=100, height=30,
            fg_color="transparent", border_width=1,
            border_color=CTK_COLORS["border"],
            text_color=CTK_COLORS["muted"], font=font_sm(),
            command=lambda: None,
        ).pack(side="left", pady=24)

        ctk.CTkFrame(self.main, height=1,
                     fg_color=CTK_COLORS["border2"]).grid(row=0, column=0, sticky="sew")

        self.content = ctk.CTkFrame(self.main, fg_color=CTK_COLORS["bg"], corner_radius=0)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self._content_frames: dict = {}
        self._build_all_content_tabs()
        self._activate_content_tab("train")

    def _build_all_content_tabs(self) -> None:
        """Build every tab once; unified_log exists immediately for mirroring."""
        for key, builder in [
            ("log",     lambda: self._build_log_tab_frame(self.content)),
            ("train",   lambda: build_train_tab(self, self.content)),
            ("apply",   lambda: build_apply_tab(self, self.content)),
            ("cluster", lambda: build_cluster_tab(self, self.content)),
        ]:
            frame = builder()
            self._content_frames[key] = frame
            frame.pack_forget()

    def _activate_content_tab(self, key: str) -> None:
        target = self._content_frames.get(key)
        for k, f in self._content_frames.items():
            if f is target:
                f.pack(fill="both", expand=True)
            else:
                f.pack_forget()

    def _clear_content(self) -> None:
        """Kept for any legacy callers; no-op in the new tab system."""
        pass

    def _update_badges(self) -> None:
        """Обновляет счётчики (badges) на кнопках Классификация и Кластеризация."""
        badges = getattr(self, "_nav_badges", {})
        if "apply" in badges:
            n = 1 if getattr(self, "apply_file", None) and self.apply_file.get() else 0
            b = badges["apply"]
            if n:
                b.configure(text=str(n), fg_color=CTK_COLORS["accent"])
                b.pack(side="right", padx=(0, 4))
            else:
                b.pack_forget()
        if "cluster" in badges:
            n = len(getattr(self, "cluster_files", []))
            b = badges["cluster"]
            if n:
                b.configure(text=str(n), fg_color=CTK_COLORS["accent"])
                b.pack(side="right", padx=(0, 4))
            else:
                b.pack_forget()

    def _switch_tab(self, key: str) -> None:
        """Переключает вкладку по ключу (train / apply / cluster / log)."""
        self._current_tab = key
        titles = {
            "train":   "Обучение модели",
            "apply":   "Классификация",
            "cluster": "Кластеризация",
            "log":     "Журнал событий",
        }
        _subtitles = {
            "train":   "Разметить данные, обучить классификатор, экспортировать модель",
            "apply":   "Пакетная классификация текстов по обученной модели",
            "cluster": "Распределение неразмеченных текстов по тематическим кластерам с LLM-неймингом",
            "log":     "Системный лог текущей сессии",
        }
        for k, b in self._nav_buttons.items():
            _active = (k == key)
            b.configure(
                fg_color="transparent",
                text_color=CTK_COLORS["fg"] if _active else CTK_COLORS["muted"],
            )
            if k in self._nav_indicators:
                self._nav_indicators[k].configure(
                    fg_color=CTK_COLORS["accent"] if _active else "transparent"
                )
        if hasattr(self, "_title"):
            self._title.configure(text=titles.get(key, key))
        if hasattr(self, "_subtitle"):
            self._subtitle.configure(text=_subtitles.get(key, ""))
        if key == "train":
            self._show_train()
        elif key == "apply":
            self._show_apply()
        elif key == "cluster":
            self._show_cluster()
        elif key == "log":
            self._show_log_tab()

    def _show_train(self) -> None:
        self._activate_content_tab("train")

    def _show_apply(self) -> None:
        self._activate_content_tab("apply")

    def _show_cluster(self) -> None:
        self._activate_content_tab("cluster")

    def _show_deps(self) -> None:
        self._open_settings_dialog()

    def _open_history_dialog(self) -> None:
        from app_dialogs_ctk import HistoryDialog
        HistoryDialog(self).show()

    def _open_artifacts_dialog(self) -> None:
        from app_dialogs_ctk import ArtifactsDialog
        ArtifactsDialog(self).show()

    def _open_settings_dialog(self) -> None:
        from app_dialogs_ctk import SettingsDialog
        SettingsDialog(self).show()

    def _show_log_tab(self) -> None:
        self._activate_content_tab("log")

    def _build_log_tab_frame(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        from ui_theme_ctk import font_mono, font_sm
        frame = ctk.CTkFrame(parent, fg_color=CTK_COLORS["bg"], corner_radius=0)
        frame.pack(fill="both", expand=True)

        toolbar = ctk.CTkFrame(frame, fg_color=CTK_COLORS["panel"], height=46, corner_radius=0)
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)
        ctk.CTkLabel(
            toolbar,
            text="Все события из вкладок Обучение / Классификация / Кластеризация",
            font=font_sm(), text_color=CTK_COLORS["muted"],
        ).pack(side="left", padx=16, pady=0)
        ctk.CTkButton(
            toolbar, text="Сохранить…", width=110,
            fg_color="transparent", border_width=1, border_color=CTK_COLORS["border2"],
            command=self._export_log,
        ).pack(side="right", padx=8, pady=8)
        ctk.CTkButton(
            toolbar, text="Очистить", width=90,
            fg_color="transparent", border_width=1, border_color=CTK_COLORS["border2"],
            command=self._clear_unified_log,
        ).pack(side="right", padx=4, pady=8)

        self.unified_log = ctk.CTkTextbox(
            frame, font=font_mono(),
            fg_color=CTK_COLORS["bg"], text_color=CTK_COLORS["muted"],
            wrap="none", state="disabled",
        )
        self.unified_log.pack(fill="both", expand=True)
        return frame

    def _clear_unified_log(self) -> None:
        ul = getattr(self, "unified_log", None)
        if ul is None:
            return
        try:
            ul.configure(state="normal")
            ul.delete("1.0", "end")
            ul.configure(state="disabled")
        except Exception:
            pass

    def _export_log(self) -> None:
        """Сохраняет содержимое unified_log в текстовый файл."""
        ul = getattr(self, "unified_log", None)
        if ul is None:
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Сохранить журнал",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="hearsy_log.txt",
        )
        if not path:
            return
        try:
            text = ul.get("1.0", "end")
            Path(path).write_text(text, encoding="utf-8")
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Ошибка", f"Не удалось сохранить журнал:\n{e}", parent=self)

    def _build_deps_tab_ctk(self, parent: ctk.CTkFrame) -> None:
        """Вкладка «Настройки»: ПК, Python-пакеты, SBERT-модели."""
        from app_train_view_ctk import Card
        from app_deps import CORE_PACKAGES, OPTIONAL_PACKAGES, _MODEL_SIZES, _MODEL_VRAM
        from config import SBERT_MODELS
        from ui_theme_ctk import font_sm, font_base, font_label, font_mono

        scroll = ctk.CTkScrollableFrame(
            parent,
            fg_color=CTK_COLORS["bg"],
            scrollbar_fg_color=CTK_COLORS["panel"],
            scrollbar_button_color=CTK_COLORS["border"],
            scrollbar_button_hover_color=CTK_COLORS["accent3"],
        )
        scroll.pack(fill="both", expand=True)

        # ── 1. Характеристики ПК ────────────────────────────────────
        hw_card = Card(scroll, title="Характеристики ПК",
                       subtitle="Определены при запуске приложения")
        hw_card.pack(fill="x", padx=20, pady=(20, 12))

        def _hw_row(label, value):
            row = ctk.CTkFrame(hw_card.body, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=label, width=120, anchor="w",
                         font=font_sm(), text_color=CTK_COLORS["muted"]).pack(side="left")
            ctk.CTkLabel(row, text=value, anchor="w",
                         font=font_base(), text_color=CTK_COLORS["fg"]).pack(side="left", padx=(4, 0))

        _hw_row("RAM:", f"{self._hw.ram_gb:.1f} ГБ")
        _hw_row("CPU:", f"{self._hw.cpu_cores} ядер")
        if self._hw.gpu_name:
            _vram = f" · VRAM {self._hw.gpu_vram_gb:.1f} ГБ" if self._hw.gpu_vram_gb else ""
            _cnt = f" (×{self._hw.gpu_count})" if self._hw.gpu_count > 1 else ""
            _hw_row("GPU:", f"{self._hw.gpu_name}{_vram}{_cnt}")
        else:
            _hw_row("GPU:", "Не обнаружен (будет использоваться CPU)")

        # ── 2. Python-пакеты ────────────────────────────────────────
        pkg_card = Card(scroll, title="Python-пакеты",
                        subtitle="Обязательные и дополнительные зависимости")
        pkg_card.pack(fill="x", padx=20, pady=(0, 12))

        self._ctk_pkg_rows: list = []

        def _make_ctk_pkg_row(pkg_parent, pip_name, import_name, install_args, desc):
            status_var = tk.StringVar(value="Проверка…")
            row = ctk.CTkFrame(pkg_parent, fg_color=CTK_COLORS["panel2"],
                               corner_radius=6)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=pip_name, width=170, anchor="w",
                         font=font_sm(), text_color=CTK_COLORS["fg"]).pack(
                side="left", padx=(10, 0), pady=4)
            status_lbl = ctk.CTkLabel(row, textvariable=status_var, width=150,
                                       anchor="w", font=font_sm(),
                                       text_color=CTK_COLORS["muted"])
            status_lbl.pack(side="left", padx=4)
            ctk.CTkLabel(row, text=desc, anchor="w", font=font_sm(),
                         text_color=CTK_COLORS["muted"]).pack(side="left", padx=4)
            btn = ctk.CTkButton(
                row, text="Установить", width=100,
                fg_color=CTK_COLORS["accent3"], hover_color=CTK_COLORS["accent"],
                font=font_sm(),
                command=lambda pn=pip_name, ia=install_args, sv=status_var, sl=status_lbl:
                    self._install_package(pn, ia, sv, None),
            )
            btn.pack(side="right", padx=8, pady=4)
            self._ctk_pkg_rows.append({
                "pip_name": pip_name, "import_name": import_name,
                "status_var": status_var, "status_lbl": status_lbl, "btn": btn,
            })

        ctk.CTkLabel(pkg_card.body, text="ОБЯЗАТЕЛЬНЫЕ", font=font_label(),
                     text_color=CTK_COLORS["muted"]).pack(anchor="w", pady=(0, 4))
        for pn, im, ia, d in CORE_PACKAGES:
            _make_ctk_pkg_row(pkg_card.body, pn, im, ia, d)

        ctk.CTkLabel(pkg_card.body, text="ДОПОЛНИТЕЛЬНЫЕ", font=font_label(),
                     text_color=CTK_COLORS["muted"]).pack(anchor="w", pady=(10, 4))
        for pn, im, ia, d in OPTIONAL_PACKAGES:
            _make_ctk_pkg_row(pkg_card.body, pn, im, ia, d)

        # Кнопки управления
        ctrl_row = ctk.CTkFrame(pkg_card.body, fg_color="transparent")
        ctrl_row.pack(fill="x", pady=(10, 0))
        ctk.CTkButton(
            ctrl_row, text="↻ Проверить статус", width=160,
            fg_color="transparent", border_width=1, border_color=CTK_COLORS["border2"],
            command=self._refresh_ctk_pkg_status,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            ctrl_row, text="Установить все недостающие", width=200,
            fg_color=CTK_COLORS["accent"], hover_color=CTK_COLORS["accent2"],
            command=getattr(self, "_check_and_install_deps", lambda: None),
        ).pack(side="left")

        # ── 3. SBERT-модели ─────────────────────────────────────────
        mdl_card = Card(scroll, title="SBERT-модели",
                        subtitle="Эмбеддинговые модели для векторизации текста")
        mdl_card.pack(fill="x", padx=20, pady=(0, 20))

        self._ctk_model_rows: list = []

        for model_id, desc in SBERT_MODELS.items():
            status_var = tk.StringVar(value="Проверка…")
            size_str = _MODEL_SIZES.get(model_id, "—")
            vram_str = _MODEL_VRAM.get(model_id, "—")
            row = ctk.CTkFrame(mdl_card.body, fg_color=CTK_COLORS["panel2"],
                               corner_radius=6)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=model_id, width=240, anchor="w",
                         font=font_sm(), text_color=CTK_COLORS["fg"]).pack(
                side="left", padx=(10, 0), pady=4)
            status_lbl = ctk.CTkLabel(row, textvariable=status_var, width=130,
                                       anchor="w", font=font_sm(),
                                       text_color=CTK_COLORS["muted"])
            status_lbl.pack(side="left", padx=4)
            ctk.CTkLabel(row, text=f"{size_str} · {vram_str} VRAM", width=160,
                         anchor="w", font=font_sm(),
                         text_color=CTK_COLORS["muted"]).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=desc[:60], anchor="w", font=font_sm(),
                         text_color=CTK_COLORS["muted"]).pack(side="left", padx=4)
            btn = ctk.CTkButton(
                row, text="Скачать", width=90,
                fg_color=CTK_COLORS["accent3"], hover_color=CTK_COLORS["accent"],
                font=font_sm(),
                command=lambda mid=model_id, sv=status_var:
                    self._download_sbert_model(mid, sv),
            )
            btn.pack(side="right", padx=8, pady=4)
            self._ctk_model_rows.append({
                "model_id": model_id, "status_var": status_var,
                "status_lbl": status_lbl, "btn": btn,
            })

        # Запустить проверку статусов в фоне
        self.after(150, self._refresh_ctk_pkg_status)

    def _refresh_ctk_pkg_status(self) -> None:
        """Проверяет статус пакетов и моделей в фоне; обновляет CTK-строки."""
        import importlib.util as _ilu

        def _run():
            try:
                import importlib.metadata as _ilm
                def _ver(pname):
                    for n in (pname, pname.lower(), pname.split()[0]):
                        try:
                            return _ilm.version(n)
                        except Exception:
                            pass
                    return None
            except ImportError:
                def _ver(_):
                    return None

            for row in getattr(self, "_ctk_pkg_rows", []):
                imp = row["import_name"]
                ok = _ilu.find_spec(imp) is not None
                ver = _ver(row["pip_name"].split()[0]) if ok else None
                sv, sl, btn = row["status_var"], row["status_lbl"], row["btn"]
                if ok:
                    self.after(0, lambda sv=sv: sv.set(f"✅ {ver or '?'}"))
                    self.after(0, lambda sl=sl: sl.configure(
                        text_color=CTK_COLORS["success"]))
                    self.after(0, lambda btn=btn: btn.configure(state="disabled"))
                else:
                    self.after(0, lambda sv=sv: sv.set("❌ Не установлен"))
                    self.after(0, lambda sl=sl: sl.configure(
                        text_color=CTK_COLORS["error"]))
                    self.after(0, lambda btn=btn: btn.configure(state="normal"))

            for row in getattr(self, "_ctk_model_rows", []):
                cached = getattr(self, "_is_model_cached",
                                 lambda _: False)(row["model_id"])
                sv, sl, btn = row["status_var"], row["status_lbl"], row["btn"]
                if cached:
                    self.after(0, lambda sv=sv: sv.set("✅ Скачана"))
                    self.after(0, lambda sl=sl: sl.configure(
                        text_color=CTK_COLORS["success"]))
                    self.after(0, lambda btn=btn: btn.configure(state="disabled"))
                else:
                    self.after(0, lambda sv=sv: sv.set("❌ Не скачана"))
                    self.after(0, lambda sl=sl: sl.configure(
                        text_color=CTK_COLORS["muted"]))
                    self.after(0, lambda btn=btn: btn.configure(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    def _minimize_win(self) -> None:
        self.iconify()

    # ---------------------------------------------------------------- tray icon
    def _start_tray(self) -> None:
        """Создаёт иконку в системном трее в фоновом потоке."""
        if not _HAS_PYSTRAY:
            return
        try:
            icon_img = self._make_tray_image("idle")
            menu = _pystray.Menu(
                _pystray.MenuItem("Показать", self._tray_show, default=True),
                _pystray.Menu.SEPARATOR,
                _pystray.MenuItem("Выход", self._tray_quit),
            )
            self._tray_icon = _pystray.Icon(
                "BankReasonTrainer",
                icon_img,
                "Bank Reason Trainer",
                menu=menu,
            )
            threading.Thread(target=self._tray_icon.run, daemon=True).start()
            self.after(1000, self._refresh_tray_status)
        except Exception as _e:
            _log.debug("tray icon creation failed, disabling tray: %s", _e)
            self._tray_icon = None

    @staticmethod
    def _make_tray_image(state: str = "idle") -> "_PILTrayImg.Image":
        """Рисует иконку трея 64×64 в зависимости от статуса."""
        size = 64
        img = _PILTrayImg.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = _PILTrayDraw.Draw(img)
        if state == "busy":
            base = (245, 190, 40, 255)   # жёлтый: есть задача
            inner = (215, 155, 25, 255)
        else:
            base = (60, 185, 95, 255)    # зелёный: ожидание
            inner = (45, 150, 75, 255)
        draw.ellipse([2, 2, 61, 61], fill=base)
        draw.ellipse([16, 16, 47, 47], fill=(255, 255, 255, 220))
        draw.ellipse([23, 23, 40, 40], fill=inner)
        return img

    def _compose_tray_status(self) -> tuple[str, str]:
        """Возвращает (state, title) для иконки трея."""
        if self._processing:
            _phase = self.train_phase.get().strip() or self.apply_phase.get().strip() or self.cluster_phase.get().strip() or "Обработка"
            _eta = self.train_eta.get().strip() or self.apply_eta.get().strip() or self.cluster_eta.get().strip() or "ETA: n/a"
            return "busy", f"Bank Reason Trainer — {_phase} | {_eta}"
        return "idle", "Bank Reason Trainer — ожидание задачи"

    def _refresh_tray_status(self) -> None:
        """Обновляет цвет и tooltip иконки трея по текущему статусу."""
        try:
            if self._tray_icon:
                _state, _title = self._compose_tray_status()
                self._tray_icon.icon = self._make_tray_image(_state)
                self._tray_icon.title = _title
        except Exception as _e:
            _log.debug("tray refresh failed: %s", _e)
        finally:
            if self.winfo_exists():
                self.after(1500, self._refresh_tray_status)

    def _tray_show(self, icon=None, item=None) -> None:
        """Показывает главное окно из трея (вызов из потока pystray)."""
        self.after(0, self._restore_from_tray)

    def _restore_from_tray(self) -> None:
        """Восстанавливает скрытое окно в главном потоке tkinter."""
        self.deiconify()
        self.lift()
        self.focus_force()

    def _tray_quit(self, icon=None, item=None) -> None:
        """Полный выход из приложения через трей."""
        if self._tray_icon:
            try:
                self._tray_icon.stop()
            except Exception as _e:
                _log.debug("tray icon stop: %s", _e)
        self._save_session()
        self.after(0, self.destroy)

    def _on_close_btn(self) -> None:
        """Обработчик кнопки ✕: скрывает в трей, или выходит если трей недоступен."""
        if _HAS_PYSTRAY and self._tray_icon:
            self.withdraw()
        else:
            self._save_session()
            self.destroy()

    # ---------------------------------------------------------- session persistence
    def _save_session(self) -> None:
        """Сохраняет текущие настройки UI в ~/.classification_tool/last_session.json."""
        try:
            data = {
                "version": 1,
                "active_tab":    ["train", "apply", "cluster", "deps"].index(self._current_tab),
                "snap":          self._snap_params(),
                "train_files":   [str(p) for p in self.train_files  if Path(p).exists()],
                "apply_file":    self.apply_file.get(),
                "model_file":    self.model_file.get(),
                "cluster_files": [str(p) for p in self.cluster_files if Path(p).exists()],
            }
            _USER_DIR.mkdir(parents=True, exist_ok=True)
            SESSION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as _e:
            _log.debug("_save_session: %s", _e)

    def _apply_snap(self, snap: dict) -> None:
        """Применяет сохранённый snap к tk.Var виджетам приложения."""
        _type_map = {
            tk.BooleanVar: bool,
            tk.IntVar:     int,
            tk.DoubleVar:  float,
            tk.StringVar:  str,
        }
        for attr, var in self.__dict__.items():
            for vtype, conv in _type_map.items():
                if isinstance(var, vtype) and attr in snap:
                    try:
                        var.set(conv(snap[attr]))
                    except Exception:
                        pass
                    break

    def _restore_session(self) -> None:
        """Тихо восстанавливает прошлую сессию из SESSION_FILE (если есть)."""
        if not SESSION_FILE.exists():
            return
        try:
            data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                return
            self._apply_snap(data.get("snap", {}))
            # — режим обучения (вычисляется из BoolVar после apply_snap) —
            if self.use_setfit.get():
                self.train_vec_mode.set("setfit")
            elif self.use_sbert_hybrid.get():
                self.train_vec_mode.set("hybrid")
            elif self.use_sbert.get():
                self.train_vec_mode.set("sbert")
            else:
                self.train_vec_mode.set("tfidf")
            # — файлы обучения —
            for p in data.get("train_files", []):
                if Path(p).exists() and p not in self.train_files:
                    self.train_files.append(p)
                    if hasattr(self, "lb_train"):
                        self.lb_train.insert("end", p)
            # — файлы кластеризации —
            for p in data.get("cluster_files", []):
                if Path(p).exists() and p not in self.cluster_files:
                    self.cluster_files.append(p)
                    if hasattr(self, "lb_cluster"):
                        self.lb_cluster.insert("end", p)
            # — apply / model —
            _af = data.get("apply_file", "")
            if _af and Path(_af).exists():
                self.apply_file.set(_af)
            _mf = data.get("model_file", "")
            if _mf and Path(_mf).exists():
                self.model_file.set(_mf)
            # — вкладка —
            _tab = int(data.get("active_tab", 0))
            _keys = ["train", "apply", "cluster", "log"]
            if 0 <= _tab < len(_keys):
                self._switch_tab(_keys[_tab])
        except Exception as _e:
            _log.debug("_restore_session: %s", _e)

    # ---------------------------------------------------------- recent files
    def _load_recents(self) -> dict:
        try:
            return json.loads(RECENTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _add_recent(self, category: str, path: str) -> None:
        data = self._load_recents()
        lst  = [p for p in data.get(category, []) if p != path][:_MAX_RECENTS - 1]
        lst.insert(0, path)
        data[category] = lst
        try:
            _USER_DIR.mkdir(parents=True, exist_ok=True)
            RECENTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as _e:
            _log.debug("_add_recent: %s", _e)

    def _show_experiment_history(self) -> None:
        """Открывает диалог истории запусков обучения."""
        try:
            from experiment_history_dialog import show_experiment_history
            show_experiment_history(self)
        except Exception as _e:
            from tkinter import messagebox as _mb
            _mb.showwarning("История", f"Не удалось открыть историю:\n{_e}", parent=self)

    def _show_recents_menu(self, anchor: tk.Widget, category: str,
                           callback: Callable[[str], None]) -> None:
        recents = self._load_recents().get(category, [])
        menu = tk.Menu(self, tearoff=0)
        if not recents:
            menu.add_command(label="(нет недавних файлов)", state="disabled")
        else:
            for p in recents:
                label = Path(p).name if len(p) > 60 else p
                menu.add_command(label=label,
                                 command=lambda _p=p: callback(_p))
        menu.tk_popup(anchor.winfo_rootx(),
                      anchor.winfo_rooty() + anchor.winfo_height())

    # ---------------------------------------------------------- user presets
    def _ask_preset_name(self) -> "str | None":
        dlg = ctk.CTkToplevel(self)
        dlg.title("Сохранить пресет")
        dlg.configure(fg_color=CTK_COLORS["bg"])
        dlg.geometry("400x150")
        dlg.resizable(False, False)
        dlg.grab_set()
        result: list = [None]
        ctk.CTkLabel(dlg, text="Название пресета:",
                     text_color=CTK_COLORS["fg"]).pack(pady=(22, 6))
        entry = ctk.CTkEntry(dlg, width=320, fg_color=CTK_COLORS["entry"],
                             text_color=CTK_COLORS["fg"], border_color=CTK_COLORS["border"])
        entry.pack()
        def _ok():
            result[0] = entry.get().strip() or None
            dlg.destroy()
        def _cancel():
            dlg.destroy()
        btns = ctk.CTkFrame(dlg, fg_color="transparent")
        btns.pack(pady=12)
        ctk.CTkButton(btns, text="OK", width=90, command=_ok).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="Отмена", width=90, fg_color="transparent",
                      border_width=1, border_color=CTK_COLORS["border2"],
                      command=_cancel).pack(side="left", padx=4)
        entry.bind("<Return>", lambda _: _ok())
        entry.bind("<Escape>", lambda _: _cancel())
        entry.focus_set()
        self.wait_window(dlg)
        return result[0]

    def _save_preset(self) -> None:
        name = self._ask_preset_name()
        if not name:
            return
        try:
            PRESETS_DIR.mkdir(parents=True, exist_ok=True)
            out = PRESETS_DIR / f"{name}.json"
            out.write_text(json.dumps(self._snap_params(), ensure_ascii=False, indent=2),
                           encoding="utf-8")
            messagebox.showinfo("Пресет сохранён", f"Пресет «{name}» сохранён.", parent=self)
        except Exception as _e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить пресет:\n{_e}", parent=self)

    def _open_preset_menu(self, anchor: tk.Widget) -> None:
        files = sorted(PRESETS_DIR.glob("*.json")) if PRESETS_DIR.exists() else []
        menu = tk.Menu(self, tearoff=0)
        if not files:
            menu.add_command(label="(нет сохранённых пресетов)", state="disabled")
        else:
            for f in files:
                menu.add_command(label=f.stem,
                                 command=lambda _f=f: self._load_preset(_f))
        menu.tk_popup(anchor.winfo_rootx(),
                      anchor.winfo_rooty() + anchor.winfo_height())

    def _load_preset(self, path: Path) -> None:
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
            self._apply_snap(snap)
        except Exception as _e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить пресет:\n{_e}", parent=self)

    def _hotkey_switch_tab(self, index: int) -> None:
        """Ctrl+1/2/3 — переключение вкладки по индексу."""
        keys = ["train", "apply", "cluster", "deps"]
        if 0 <= index < len(keys):
            self._switch_tab(keys[index])

    def _run_active_tab(self) -> None:
        """F5/Ctrl+Enter — запускает операцию на активной вкладке."""
        handlers = {
            "train":   getattr(self, "run_training", None),
            "apply":   getattr(self, "run_apply",    None),
            "cluster": getattr(self, "run_cluster",  None),
        }
        h = handlers.get(self._current_tab)
        if h:
            h()

    def _register_sub_tabs(self, main_idx: int, labels: list, frames: list,
                           default: int = 0) -> None:
        """No-op: sub-tabs were removed in CTK migration."""

    def _switch_sub_tab(self, main_idx: int, sub_idx: int) -> None:
        """No-op: sub-tabs were removed in CTK migration."""

    def _toggle_help_panel(self) -> None:
        """No-op: right-panel help removed in CTK migration."""

    def _apply_gpu_optimal_params(self) -> None:
        """Применяет оптимальные параметры GPU сразу после инициализации UI."""
        hw = self._hw
        if hw.gpu_name and hw.gpu_vram_gb is not None:
            self._apply_all_hw_params()

    def _apply_all_hw_params(self) -> None:
        """Применяет оптимальные параметры для всех режимов по профилю железа.

        Вызывается: при старте приложения (если есть GPU) и при нажатии
        кнопки «Определить конфигурацию ПК».
        """
        hw = self._hw

        # --- Batch sizes ---
        self.sbert_batch.set(hw.sbert_batch)
        self.setfit_batch.set(hw.setfit_batch)
        self.t5_batch_size.set(hw.t5_batch)
        self.max_features.set(hw.max_features)

        # --- GPU toggle + device ---
        if hw.gpu_name and hw.gpu_vram_gb is not None:
            self.use_gpu_all.set(True)
            self._on_gpu_all_toggle()

        # --- TF32 для Ampere+ (A100, RTX 30xx/40xx): ~20-30% ускорение matmul ---
        if hw.gpu_supports_tf32:
            try:
                import torch
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except (ImportError, AttributeError) as _e:
                _log.debug("TF32 enable failed: %s", _e)

    def _log_gpu_startup(self) -> None:
        """Выводит информацию о GPU в unified_log при старте приложения."""
        hw = self._hw
        if hw.gpu_name and hw.gpu_vram_gb is not None:
            if hw.gpu_count >= 2:
                _names = "  +  ".join(hw.gpu_names)
                _header = (
                    f"[GPU] {hw.gpu_count}× GPU: {_names} "
                    f"({hw.gpu_vram_gb:.0f} ГБ × {hw.gpu_count}) — CUDA активна ✅"
                )
            else:
                _header = (
                    f"[GPU] {hw.gpu_name} ({hw.gpu_vram_gb:.0f} ГБ VRAM) — CUDA активна ✅"
                )
            msg = (
                f"{_header}  |  "
                f"SBERT-батч: {hw.sbert_batch}  |  "
                f"T5-батч: {hw.t5_batch}  |  "
                f"SetFit-батч: {hw.setfit_batch}"
            )
            tag = "success"
        else:
            msg = "[GPU] CUDA не обнаружена — все задачи выполняются на CPU"
            tag = "dim"
        # Вывести в train_log если он уже создан
        try:
            log_widget = getattr(self, "train_log", None)
            if log_widget is not None:
                self._log_to(log_widget, msg)
            else:
                _log.info("GPU startup: %s", msg)
        except Exception as _e:
            _log.debug("gpu startup log: %s", _e)

    def _toggle_right_log(self) -> None:
        """No-op: right-panel removed in CTK migration."""

    # ------------------------------------------------ action block + log section
    def _build_action_block(
        self,
        parent: tk.Widget,
        btn_text: str,
        btn_cmd: Callable,
        progress_var: tk.DoubleVar,
        pct_var: tk.StringVar,
        phase_var: tk.StringVar,
        speed_var: tk.StringVar,
        eta_var: tk.StringVar,
        label: str = "Прогресс",
    ) -> Tuple["RoundedButton", ttk.Button]:
        """Создаёт панель запуска + кнопку стоп + прогресс-бар + детальный статус (2 строки)."""
        outer = ttk.LabelFrame(parent, text=label, padding=10)
        outer.pack(fill="x", pady=(0, 10))

        # Строка 1: кнопка + стоп + прогресс-бар + процент
        row0 = ttk.Frame(outer, style="Card.TFrame")
        row0.pack(fill="x")
        btn = RoundedButton(row0, text=btn_text, command=btn_cmd, bg=PANEL)
        btn.pack(side="left", pady=2)
        stop_btn = ttk.Button(
            row0, text="⏹ Стоп", command=self._request_cancel,
            state="disabled", width=8,
        )
        stop_btn.pack(side="left", padx=(6, 0), pady=2)
        Tooltip(stop_btn, "Остановить текущую операцию после завершения текущего шага.\nДанные, обработанные до этого момента, сохраняются.")
        ttk.Progressbar(row0, variable=progress_var, maximum=100.0).pack(
            side="left", fill="x", expand=True, padx=12
        )
        ttk.Label(row0, textvariable=pct_var, width=5, anchor="e",
                  style="Card.Muted.TLabel").pack(side="right")

        # Строка 2: фаза | скорость | ETA
        row1 = ttk.Frame(outer, style="Card.TFrame")
        row1.pack(fill="x", pady=(6, 0))
        ttk.Label(row1, textvariable=phase_var,
                  style="Card.Muted.TLabel", anchor="w").pack(side="left")
        sep1 = ttk.Separator(row1, orient="vertical")
        sep1.pack(side="left", padx=10, fill="y")
        ttk.Label(row1, text="⚡", style="Card.Muted.TLabel").pack(side="left")
        ttk.Label(row1, textvariable=speed_var,
                  style="Card.Muted.TLabel", anchor="w").pack(side="left", padx=(2, 0))
        sep2 = ttk.Separator(row1, orient="vertical")
        sep2.pack(side="left", padx=10, fill="y")
        ttk.Label(row1, text="⏱", style="Card.Muted.TLabel").pack(side="left")
        ttk.Label(row1, textvariable=eta_var,
                  style="Card.Muted.TLabel", anchor="w").pack(side="left", padx=(2, 0))
        return btn, stop_btn

    def _request_cancel(self):
        """Запрашивает мягкую отмену текущей операции (проверяется в worker-ах на чекпоинтах)."""
        self._cancel_event.set()

    def _build_log_section(
        self,
        parent: tk.Widget,
        log_attr: str,
        height: int = 8,
        expand: bool = False,
    ) -> tk.Text:
        """Создаёт тулбар управления логом + текстовый виджет.
        Сохраняет виджет как self.<log_attr> для совместимости с log_train/apply/cluster."""
        toolbar = ttk.Frame(parent, style="Card.TFrame", padding=(4, 3))
        toolbar.pack(fill="x")

        ttk.Label(toolbar, text="Лог", style="Card.TLabel",
                  font=("Segoe UI", 9, "bold")).pack(side="left", padx=(2, 8))

        txt = tk.Text(
            parent, height=height, wrap="word",
            bg=ENTRY_BG, fg=FG, insertbackground=FG,
            selectbackground=SELECT, selectforeground=FG,
            relief="flat", padx=8, pady=6,
            font=("Segoe UI", 10),
        )

        def _clear():
            txt.configure(state="normal")
            txt.delete("1.0", "end")

        def _save():
            content = txt.get("1.0", "end").strip()
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Текст", "*.txt"), ("Все файлы", "*.*")],
                title="Сохранить лог",
            )
            if path:
                try:
                    Path(path).write_text(content, encoding="utf-8")
                except Exception as ex:
                    messagebox.showerror("Ошибка сохранения", str(ex))

        def _copy():
            content = txt.get("1.0", "end").strip()
            self.clipboard_clear()
            self.clipboard_append(content)

        def _end():
            txt.see("end")

        btn_end = ttk.Button(toolbar, text="↓", width=2, command=_end)
        btn_end.pack(side="right", padx=2)
        Tooltip(btn_end, "Прокрутить в конец лога")
        btn_copy = ttk.Button(toolbar, text="📋 Копировать", command=_copy)
        btn_copy.pack(side="right", padx=2)
        Tooltip(btn_copy, "Скопировать весь лог в буфер обмена")
        btn_save = ttk.Button(toolbar, text="💾 Сохранить", command=_save)
        btn_save.pack(side="right", padx=2)
        Tooltip(btn_save, "Сохранить лог в текстовый файл (.txt)")
        btn_clear = ttk.Button(toolbar, text="🗑 Очистить", command=_clear)
        btn_clear.pack(side="right", padx=2)
        Tooltip(btn_clear, "Очистить содержимое лога")

        txt.pack(fill="both" if expand else "x", expand=expand, pady=(0, 8))
        setattr(self, log_attr, txt)
        return txt

    # ----------------------------------------------------------- UI helpers

    def _build_readiness_bar(
        self,
        parent: tk.Widget,
        checks: "List[Tuple[Callable[[], Tuple[bool, str]], ...]]",
    ) -> None:
        """Строка готовности: бейджи с авто-обновлением каждые 400 мс.

        checks: List[Callable[[], Tuple[bool, str]]]
            каждая fn() возвращает (ok, display_text)
        """
        _fnt = (_best_font(), 9)
        _fnt_bold = (_best_font(), 9, "bold")
        bar = tk.Frame(parent, bg=PANEL2)
        bar.pack(fill="x", pady=(0, 6))

        lbl_list = []
        for i, check_fn in enumerate(checks):
            if i > 0:
                tk.Label(bar, text="  ·  ", bg=PANEL2, fg=MUTED2, font=_fnt).pack(side="left")
            lbl = tk.Label(bar, text="…", bg=PANEL2, fg=MUTED, font=_fnt)
            lbl.pack(side="left", padx=(6 if i == 0 else 0, 0))
            lbl_list.append((check_fn, lbl))

        ready_lbl = tk.Label(bar, text="…", bg=PANEL2, fg=MUTED, font=_fnt_bold)
        ready_lbl.pack(side="right", padx=(0, 8))

        def _update() -> None:
            if not bar.winfo_exists():
                return
            all_ok = True
            for fn, lbl in lbl_list:
                try:
                    ok, text = fn()
                except Exception:
                    ok, text = False, "?"
                lbl.configure(text=text, fg=SUCCESS if ok else WARNING)
                if not ok:
                    all_ok = False
            ready_lbl.configure(
                text="✅ Готово к запуску" if all_ok else "⚠ Укажите все параметры",
                fg=SUCCESS if all_ok else WARNING,
            )
            bar.after(400, _update)

        bar.after(150, _update)

    def _combobox(self, parent, row: int, label: str, var: tk.StringVar, tooltip: str) -> ttk.Combobox:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=(6, 0) if row else (0, 0))
        cb = ttk.Combobox(parent, textvariable=var, state="readonly", width=64, values=[])
        cb.grid(row=row, column=1, sticky="w", padx=10, pady=(6, 0) if row else (0, 0))
        self.attach_help(cb, f"Колонка: {label}",
                         f"Выбор колонки Excel для поля «{label}».\n"
                         f"Поля могут быть пустыми.\n"
                         f"Если поле часто пустое — включи Auto profile per row или снизь вес.",
                         tooltip)
        return cb

    def _weight_slider(self, parent, row: int, label: str, var: tk.IntVar) -> tk.Scale:
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", pady=(6, 0) if row else (0, 0))
        sc = tk.Scale(parent, from_=0, to=5, orient="horizontal", variable=var, length=340,
                      bg=BG, fg=FG, troughcolor=ENTRY_BG, activebackground=ACCENT,
                      highlightthickness=0, bd=0, resolution=1)
        sc.grid(row=row, column=1, sticky="w", padx=10, pady=(6, 0) if row else (0, 0))
        self.attach_help(sc, f"Вес: {label}",
                         "Вес секции = насколько сильно она влияет на модель.\n"
                         "Технически: секция повторяется во входном тексте X (0..5).\n\n"
                         "Рекомендация:\n"
                         "- Клиент: 2–3\n- Оператор: 0–1\n- Summary: 0–5 (если качественная)\n"
                         "- Описание: 1–2\n- Ответы: 0–2",
                         "0=не учитывать, 5=максимум")
        return sc

    # ------------------------------------------------------ combobox refresh
    def _refresh_combobox_values(self, headers: List[str]):
        self._headers_cache = headers[:]
        vals = [""] + headers

        # Комбобоксы, которые НЕ должны получать заголовки Excel-колонок.
        # Сравниваем по Tk-пути виджета (str), а не по id Python-объекта —
        # это надёжнее при вложении через Canvas (ScrollableFrame.inner).
        _protected_names: set = set()
        for _attr in (
            "cb_auto_profile",
            "cb_sbert_model_combo",
            "cb_sbert_device_combo",
            "cb_sbert_clust_model",
            "cb_sbert_clust_device",
            "cb_pred",
            "cb_calib_method",
            "cb_umap_metric",
            "cb_setfit_model_combo",
            "cb_llm_provider",
        ):
            _w = getattr(self, _attr, None)
            if _w is not None:
                _protected_names.add(str(_w))

        def walk(w):
            out = []
            for ch in w.winfo_children():
                if isinstance(ch, ttk.Combobox):
                    out.append(ch)
                out.extend(walk(ch))
            return out

        for cb in walk(self):
            if str(cb) in _protected_names:
                continue
            try:
                cb.configure(values=vals)
            except Exception as _e:
                _log.debug("combobox configure(values): %s", _e)

        if hasattr(self, "cb_pred"):
            self.cb_pred.configure(values=headers + ["pred_marker1", "pred_label", "pred_reason"])

        self._auto_pick_if_possible()

    def _pick_by_candidates(self, var: tk.StringVar, candidates: List[str], headers: List[str]):
        cur = (var.get() or "").strip()
        if cur and cur in headers:
            return
        for c in candidates:
            if c in headers:
                var.set(c)
                return

    def _auto_pick_if_possible(self):
        h = self._headers_cache
        if not h:
            return
        self._pick_by_candidates(self.desc_col, DEFAULT_COLS["desc"], h)
        self._pick_by_candidates(self.call_col, DEFAULT_COLS["call"], h)
        self._pick_by_candidates(self.chat_col, DEFAULT_COLS["chat"], h)
        self._pick_by_candidates(self.summary_col, DEFAULT_COLS["summary"], h)
        self._pick_by_candidates(self.ans_short_col, DEFAULT_COLS["ans_short"], h)
        self._pick_by_candidates(self.ans_full_col, DEFAULT_COLS["ans_full"], h)
        self._pick_by_candidates(self.label_col, DEFAULT_COLS["label"], h)

    # ----------------------------------------------------------- GPU master switch
    def _on_gpu_all_toggle(self, *_):
        """Переключатель «Весь GPU»: синхронизирует sbert_device и состояние комбобоксов."""
        enabled = bool(self.use_gpu_all.get())
        self.sbert_device.set("cuda" if enabled else "auto")
        # Блокируем/разблокируем комбобоксы выбора устройства на всех вкладках
        state = "disabled" if enabled else "readonly"
        for attr in ("cb_sbert_device_combo", "cb_sbert_clust_device"):
            cb = getattr(self, attr, None)
            if cb is not None:
                try:
                    cb.configure(state=state)
                except Exception as _e:
                    _log.debug("GPU combo state(%s): %s", attr, _e)
        self._update_guardrails()

    # ----------------------------------------------------------- guardrails
    def _bind_guardrails(self):
        for v in [
            self.w_desc, self.w_client, self.w_operator, self.w_summary, self.w_ans_short, self.w_ans_full,
            self.char_ng_min, self.char_ng_max, self.word_ng_min, self.word_ng_max,
            self.min_df, self.max_features, self.C, self.test_size,
            self.use_summary, self.auto_profile, self.train_mode, self.base_model_file,
            self.use_stop_words, self.use_noise_tokens, self.use_noise_phrases,
            self.sublinear_tf, self.n_init_cluster, self.use_elbow,
            self.k_clusters, self.use_sbert, self.sbert_model, self.sbert_device,
            self.use_gpu_all,
        ]:
            try:
                v.trace_add("write", lambda *_: self._update_guardrails())
            except Exception as _e:
                _log.debug("trace_add(guardrails): %s", _e)
        self.use_gpu_all.trace_add("write", self._on_gpu_all_toggle)
        self._update_guardrails()

    def _update_guardrails(self):
        warns = []
        if int(self.w_client.get()) == 0:
            warns.append("⚠️ Клиент=0: модель почти не видит намерение клиента.")
        if int(self.w_operator.get()) > int(self.w_client.get()) and int(self.w_client.get()) > 0:
            warns.append("⚠️ Оператор > Клиент: риск доминирования шаблонов оператора.")

        cmin, cmax = int(self.char_ng_min.get()), int(self.char_ng_max.get())
        if cmin > cmax:
            warns.append("⚠️ char_ngram: min > max.")
        if cmax >= 8:
            warns.append("⚠️ char_ngram max >= 8: может стать тяжело по RAM/времени.")

        wmin, wmax = int(self.word_ng_min.get()), int(self.word_ng_max.get())
        if wmin > wmax:
            warns.append("⚠️ word_ngram: min > max.")
        if wmax >= 4:
            warns.append("⚠️ word_ngram max >= 4: может дать шум/переобучение + ресурсы.")

        if int(self.min_df.get()) >= 6:
            warns.append("⚠️ min_df высокий: режет редкие причины/термины.")
        try:
            mf = int(float(self.max_features.get()))
            if mf >= 800000:
                warns.append("⚠️ max_features >= 800k: риск RAM.")
        except Exception:
            warns.append("⚠️ max_features должно быть числом.")

        try:
            c_val = float(self.C.get())
            if c_val <= 0:
                warns.append("⚠️ C должно быть > 0 (LinearSVC требует положительный параметр регуляризации).")
        except Exception:
            warns.append("⚠️ C: введите число > 0.")

        try:
            mi_val = int(self.max_iter.get())
            if mi_val <= 0:
                warns.append("⚠️ max_iter должно быть >= 1.")
        except Exception:
            warns.append("⚠️ max_iter: введите целое число >= 1.")

        try:
            ts = float(self.test_size.get())
            if ts < 0:
                warns.append("⚠️ test_size < 0: недопустимое значение — используйте 0 (без валидации) или 0.1–0.3.")
            elif ts >= 1.0:
                warns.append("⚠️ test_size >= 1.0: недопустимое значение — должно быть от 0 до 1 (не включая 1).")
            elif ts >= 0.35:
                warns.append("⚠️ test_size >= 0.35: слишком много в holdout (особенно на малых данных).")
        except Exception:
            warns.append("⚠️ test_size: введите число от 0 до 0.5.")

        if self.train_mode.get() == "finetune" and not self.base_model_file.get().strip():
            warns.append("⚠️ Дообучение: выбери базовую модель .joblib.")

        if self.use_elbow.get() and (int(self.k_clusters.get()) > 80):
            warns.append("⚠️ Elbow+K>80: может быть долго. Реком. K<=60 для подбора.")

        if self.use_sbert.get() and not SBERTVectorizer.is_available():
            warns.append("⚠️ SBERT: sentence-transformers недоступен.\n"
                         "   Возможные причины: не установлен, или DLL-конфликт torchvision.\n"
                         "   Нажмите «Установить» в разделе SBERT,\n"
                         "   или нажмите «⚡ Установить torch+CUDA» для пересборки torch-стека.")
        if self.sbert_device.get() == "cuda" and not SBERTVectorizer.is_cuda_available():
            warns.append("⚠️ SBERT device=cuda: CUDA недоступна (нет GPU или torch-cpu).\n   Используйте cpu или auto.")

        if self.use_gpu_all.get() and not SBERTVectorizer.is_cuda_available():
            warns.append("⚠️ «GPU для всех секций»: CUDA недоступна.\n"
                         "   Установите PyTorch с CUDA или отключите переключатель.")

        self.set_warnings("\n".join(warns))

    # --------------------------------------------------- feature text (apply/cluster)
    def _row_to_feature_text(
        self,
        row_vals: List[Any],
        header: List[str],
        snap: Dict[str, Any],
        header_index: Optional[Dict[str, int]] = None,
    ) -> str:
        """Unified feature building for apply and cluster (avoids duplicate logic).

        Reads ALL UI state from *snap* (captured in main thread before worker starts)
        so this method is safe to call from background threads.
        """
        def get(i):
            return row_vals[i] if (i is not None and i < len(row_vals)) else None

        index_map = header_index or {col: i for i, col in enumerate(header)}

        def resolve_idx(col_name: str) -> Optional[int]:
            i = index_map.get(col_name)
            if i is None:
                i = idx_of(header, col_name)
            return i

        i_desc = resolve_idx(snap["desc_col"])
        i_call = resolve_idx(snap["call_col"])
        i_chat = resolve_idx(snap["chat_col"])
        i_sum  = resolve_idx(snap["summary_col"]) if snap["use_summary"] else None
        i_as   = resolve_idx(snap["ans_short_col"])
        i_af   = resolve_idx(snap["ans_full_col"])

        desc = normalize_text(get(i_desc)) if i_desc is not None else ""
        call_raw = get(i_call) if i_call is not None else None
        chat_raw = get(i_chat) if i_chat is not None else None
        summ = normalize_text(get(i_sum)) if i_sum is not None else ""
        ans_s = clean_answer_text(get(i_as)) if i_as is not None else ""
        ans_f = clean_answer_text(get(i_af)) if i_af is not None else ""

        ign_bot = snap["ignore_chatbot"]
        call_clean, call_client, call_oper, r1 = parse_dialog_roles(call_raw, ignore_chatbot=ign_bot)
        chat_clean, chat_client, chat_oper, r2 = parse_dialog_roles(chat_raw, ignore_chatbot=ign_bot)
        roles_found = bool(r1 or r2)

        client_text = "\n".join([t for t in [call_client, chat_client] if t]).strip()
        operator_text = "\n".join([t for t in [call_oper, chat_oper] if t]).strip()

        has_dialog = bool(call_clean or chat_clean)
        channel = (
            "call" if (call_clean and not chat_clean) else
            "chat" if (chat_clean and not call_clean) else
            "call+chat" if (call_clean and chat_clean) else
            "none"
        )

        weights = choose_row_profile_weights(
            base=snap["base_w"],
            auto_profile=snap["auto_profile"],
            has_desc=bool(desc),
            has_dialog=has_dialog,
            roles_found=roles_found,
            has_summary=bool(summ),
            has_ans_s=bool(ans_s),
            has_ans_f=bool(ans_f),
        )

        if not any([desc, client_text, operator_text, summ, ans_s, ans_f]):
            return ""  # all content fields empty → skip row in apply/cluster

        return build_feature_text(
            channel=channel,
            desc=desc,
            client_text=client_text,
            operator_text=operator_text,
            summary=summ,
            ans_short=ans_s,
            ans_full=ans_f,
            weights=weights,
            normalize_entities=bool(snap.get("use_entity_norm", False)) if snap else False,
        )

    # ------------------------------------------------------------------ log
    def _log_to(self, widget, msg: str) -> None:
        """Универсальный helper: добавляет строку в лог-виджет.
        ВСЕГДА вызывать из главного потока (или через self.after(0, ...)).
        Корректно обрабатывает виджеты в state='disabled'."""
        # CTkTextbox.cget("state") raises ValueError — proxy to inner _textbox if needed
        _tw = getattr(widget, "_textbox", widget)
        was_disabled = str(_tw.cget("state")) == "disabled"
        if was_disabled:
            widget.configure(state="normal")
        widget.insert("end", msg + "\n")
        widget.see("end")
        if was_disabled:
            widget.configure(state="disabled")
        # Зеркалировать в единый лог (если уже создан как CTkTextbox)
        ul = getattr(self, "unified_log", None)
        if ul is not None and ul is not widget:
            try:
                ul.configure(state="normal")
                ul.insert("end", msg + "\n")
                ul.see("end")
                ul.configure(state="disabled")
            except Exception:
                pass

    # --------------------------------------------------------- config → UI
    def _apply_config_to_ui(self, cfg: Dict[str, Any]):
        self.desc_col.set(cfg.get("desc_col", self.desc_col.get()))
        self.call_col.set(cfg.get("call_col", self.call_col.get()))
        self.chat_col.set(cfg.get("chat_col", self.chat_col.get()))
        self.summary_col.set(cfg.get("summary_col", self.summary_col.get()))
        self.ans_short_col.set(cfg.get("answer_short_col", self.ans_short_col.get()))
        self.ans_full_col.set(cfg.get("answer_full_col", self.ans_full_col.get()))
        self.label_col.set(cfg.get("label_col", self.label_col.get()))

        self.use_summary.set(bool(cfg.get("use_summary", self.use_summary.get())))
        self.ignore_chatbot.set(bool(cfg.get("ignore_chatbot", self.ignore_chatbot.get())))
        self.auto_profile.set(cfg.get("auto_profile", self.auto_profile.get()))
        self.use_stop_words.set(bool(cfg.get("use_stop_words", self.use_stop_words.get())))
        self.use_noise_tokens.set(bool(cfg.get("use_noise_tokens", self.use_noise_tokens.get())))
        self.use_noise_phrases.set(bool(cfg.get("use_noise_phrases", self.use_noise_phrases.get())))
        self.sublinear_tf.set(bool(cfg.get("sublinear_tf", self.sublinear_tf.get())))

        for k, var in [
            ("w_desc", self.w_desc), ("w_client", self.w_client), ("w_operator", self.w_operator),
            ("w_summary", self.w_summary), ("w_answer_short", self.w_ans_short), ("w_answer_full", self.w_ans_full)
        ]:
            if k in cfg:
                var.set(int(cfg[k]))

        for k, var in [
            ("char_ng_min", self.char_ng_min), ("char_ng_max", self.char_ng_max),
            ("word_ng_min", self.word_ng_min), ("word_ng_max", self.word_ng_max),
            ("min_df", self.min_df), ("max_features", self.max_features),
            ("C", self.C), ("max_iter", self.max_iter), ("test_size", self.test_size)
        ]:
            if k in cfg:
                try:
                    var.set(cfg[k])
                except Exception as _e:
                    _log.debug("config load var.set(%s): %s", k, _e)

        self.class_weight_balanced.set(bool(cfg.get("class_weight_balanced", self.class_weight_balanced.get())))
        ct = cfg.get("classifier_type", "")
        if ct:
            self.set_help("Classifier", f"Модель использует: {ct}")

        # Флаги шума и векторайзера
        self.use_per_field.set(bool(cfg.get("use_per_field", True)))
        self.use_svd.set(bool(cfg.get("use_svd", False)))
        try:
            self.svd_components.set(int(cfg.get("svd_components", 300)))
        except Exception as _e:
            _log.debug("config load svd_components: %s", _e)
        self.use_lemma.set(bool(cfg.get("use_lemma", False)))

        # SBERT
        self.use_sbert.set(bool(cfg.get("use_sbert", False)))
        self.use_sbert_hybrid.set(bool(cfg.get("use_sbert_hybrid", False)))
        sbert_m = cfg.get("sbert_model", SBERT_DEFAULT)
        if sbert_m in SBERT_MODELS_LIST:
            self.sbert_model.set(sbert_m)
        elif sbert_m:
            self.sbert_model.set(sbert_m)  # кастомное имя
        # Мета-признаки
        self.use_meta.set(bool(cfg.get("use_meta", False)))
        # SMOTE балансировка
        self.use_smote.set(bool(cfg.get("use_smote", True)))
        self.diagnostic_mode.set(bool(cfg.get("diagnostic_mode", self.diagnostic_mode.get())))
        # Метод калибровки
        _cm = cfg.get("calib_method", "sigmoid")
        if _cm in ("auto", "sigmoid", "isotonic"):
            self.calib_method.set(_cm)

    # ---------------------------------------------------------- weights dict
    def _weights_dict(self) -> Dict[str, int]:
        return {
            "w_desc":         int(self.w_desc.get()),
            "w_client":       int(self.w_client.get()),
            "w_operator":     int(self.w_operator.get()),
            "w_summary":      int(self.w_summary.get()) if self.use_summary.get() else 0,
            "w_answer_short": int(self.w_ans_short.get()),
            "w_answer_full":  int(self.w_ans_full.get()),
        }

    def _snap_params(self) -> Dict[str, Any]:
        """Снимает ВСЕ параметры UI в dict.

        Вызывать ТОЛЬКО из главного потока, ДО запуска рабочего потока.
        Передавать снимок (а не self) в worker, чтобы избежать race conditions
        при чтении tk.Var из фонового потока.
        """
        try:
            max_features_int = int(self.max_features.get())
        except Exception:
            max_features_int = 100_000

        return {
            # — колонки —
            "desc_col":       self.desc_col.get().strip(),
            "call_col":       self.call_col.get().strip(),
            "chat_col":       self.chat_col.get().strip(),
            "summary_col":    self.summary_col.get().strip(),
            "ans_short_col":  self.ans_short_col.get().strip(),
            "ans_full_col":   self.ans_full_col.get().strip(),
            "label_col":      self.label_col.get().strip(),
            # — поведение —
            "use_summary":    bool(self.use_summary.get()),
            "ignore_chatbot": bool(self.ignore_chatbot.get()),
            "auto_profile":   (self.auto_profile.get() or "off").strip(),
            # — веса —
            "base_w":         self._weights_dict(),
            "w_desc":         int(self.w_desc.get()),
            "w_client":       int(self.w_client.get()),
            "w_operator":     int(self.w_operator.get()),
            "w_summary":      int(self.w_summary.get()) if self.use_summary.get() else 0,
            "w_ans_short":    int(self.w_ans_short.get()),
            "w_ans_full":     int(self.w_ans_full.get()),
            # — векторайзер —
            "char_ng":        (int(self.char_ng_min.get()), int(self.char_ng_max.get())),
            "word_ng":        (int(self.word_ng_min.get()), int(self.word_ng_max.get())),
            "min_df":         int(self.min_df.get()),
            "max_features":   max_features_int,
            "sublinear_tf":   bool(self.sublinear_tf.get()),
            "use_stop_words":    bool(self.use_stop_words.get()),
            "use_noise_tokens":  bool(self.use_noise_tokens.get()),
            "use_noise_phrases": bool(self.use_noise_phrases.get()),
            "extra_stop_words":  list(self._user_exclusions.get("stop_words", [])),
            "extra_noise_tokens":  list(self._user_exclusions.get("noise_tokens", [])),
            "extra_noise_phrases": list(self._user_exclusions.get("noise_phrases", [])),
            "use_per_field":     bool(self.use_per_field.get()),
            "use_svd":           bool(self.use_svd.get()),
            "svd_components":    max(50, int(self.svd_components.get())),
            "use_lemma":         bool(self.use_lemma.get()),
            "use_meta":          bool(self.use_meta.get()),
            "use_sbert_hybrid":  bool(self.use_sbert_hybrid.get()),
            "use_smote":         bool(self.use_smote.get()),
            "oversample_strategy": (self.oversample_strategy.get().strip() or "augment_light"),
            "drop_conflicts":    bool(self.drop_conflicts.get()),
            "use_llm_augment":       bool(self.use_llm_augment.get()),
            "augment_min_samples":   max(5, int(self.augment_min_samples.get())),
            "augment_n_paraphrases": max(1, min(10, int(self.augment_n_paraphrases.get()))),
            "detect_near_dups":      bool(self.detect_near_dups.get()),
            "near_dup_threshold":    max(0.70, min(0.99, float(self.near_dup_threshold.get()))),
            "use_hard_negatives":    bool(self.use_hard_negatives.get()),
            "use_field_dropout":     bool(self.use_field_dropout.get()),
            "field_dropout_prob":    max(0.05, min(0.50, float(self.field_dropout_prob.get()))),
            "field_dropout_copies":  max(1, min(5, int(self.field_dropout_copies.get()))),
            "use_entity_norm":       bool(self.use_entity_norm.get()),
            "detect_mislabeled":     bool(self.detect_mislabeled.get()),
            "mislabeled_threshold":  max(0.05, min(0.60, float(self.mislabeled_threshold.get()))),
            "use_pseudo_label":      bool(self.use_pseudo_label.get()),
            "pseudo_label_file":     self.pseudo_label_file.get().strip(),
            "pseudo_label_threshold": max(0.70, min(0.99, float(self.pseudo_label_threshold.get()))),
            "use_hierarchical":      bool(self.use_hierarchical.get()),
            "use_anchor_texts":      bool(self.use_anchor_texts.get()),
            "anchor_copies":         max(1, min(10, int(self.anchor_copies.get()))),
            "use_confident_learning":          bool(self.use_confident_learning.get()),
            "confident_learning_threshold":    max(0.1, min(2.0, float(self.confident_learning_threshold.get()))),
            "use_kfold_ensemble":    bool(self.use_kfold_ensemble.get()),
            "kfold_k":               max(2, min(10, int(self.kfold_k.get()))),
            "use_optuna":            bool(self.use_optuna.get()),
            "n_optuna_trials":       max(5, min(100, int(self.n_optuna_trials.get()))),
            # — классификатор —
            "C":              float(self.C.get()),
            "max_iter":       int(self.max_iter.get()),
            "balanced":       bool(self.class_weight_balanced.get()),
            "calib_method":   self.calib_method.get() or "sigmoid",
            "test_size":      float(self.test_size.get()),
            # — SBERT —
            "use_sbert":      bool(self.use_sbert.get()),
            "sbert_model":    self.sbert_model.get().strip() or SBERT_DEFAULT,
            "sbert_model2":   self.sbert_model2.get().strip() or SBERT_DEFAULT,
            "ensemble_vec2":  self.ensemble_vec2.get().strip() or "tfidf",
            "sbert_device":   self.sbert_device.get().strip() or "auto",
            "sbert_batch":    max(1, int(self.sbert_batch.get())),
            # — глобальный GPU-переключатель —
            "use_gpu_all":    bool(self.use_gpu_all.get()),
            # — SetFit нейросетевой классификатор —
            "use_setfit":           bool(self.use_setfit.get()),
            "setfit_model":         self.setfit_model.get().strip() or SETFIT_DEFAULT,
            "setfit_epochs":        max(1, int(self.setfit_epochs.get())),
            "setfit_num_iterations": max(5, int(self.setfit_num_iterations.get())),
            "setfit_batch":         max(1, int(self.setfit_batch.get())),
            "setfit_fp16":          bool(self.setfit_fp16.get()),
            # — авто-параметры ПК (не в UI, но сохраняются в snap) —
            "chunk":          self._hw.chunk,
            "kmeans_batch":   self._hw.kmeans_batch,
            # — режим интерфейса (Авто / Ручной) —
            "_expert_mode_var": bool(getattr(self, "_expert_mode_var", tk.BooleanVar(value=False)).get()),
            # — режим обучения —
            "train_mode":     self.train_mode.get(),
            "base_model_file": self.base_model_file.get().strip(),
            # — apply / ensemble —
            "use_ensemble":    bool(self.use_ensemble.get()),
            "ensemble_model2": self.ensemble_model2.get().strip(),
            "ensemble_w1":     max(0.05, min(0.95, float(self.ensemble_w1.get()))),
            "pred_col":       self.pred_col.get().strip() or "pred_marker1",
            "review_threshold": float(self.review_threshold.get()),
            "use_other_label":       bool(self.use_other_label.get()),
            "other_label_threshold": float(self.other_label_threshold.get()),
            "other_label_text":      self.other_label_text.get().strip() or DEFAULT_OTHER_LABEL,
            "use_per_class_other_threshold": bool(self.use_per_class_other_threshold.get()),
            "use_custom_class_thresholds":   bool(self.use_custom_class_thresholds.get()),
            "use_llm_rerank":    bool(self.use_llm_rerank.get()),
            "llm_rerank_low":    max(0.0, min(0.9, float(self.llm_rerank_low.get()))),
            "llm_rerank_high":   max(0.1, min(0.95, float(self.llm_rerank_high.get()))),
            "llm_rerank_top_k":  max(2, min(5, int(self.llm_rerank_top_k.get()))),
            "use_label_smoothing":   bool(self.use_label_smoothing.get()),
            "label_smoothing_eps":   max(0.0, min(0.3, float(self.label_smoothing_eps.get()))),
            "use_pos_tags":          bool(self.use_pos_tags.get()),
            "use_ambiguity_detector": bool(self.use_ambiguity_detector.get()),
            "ambiguity_epsilon":      max(0.01, min(0.30, float(self.ambiguity_epsilon.get()))),
            # — кластеризация —
            "k_clusters":       int(self.k_clusters.get()),
            "use_elbow":        bool(self.use_elbow.get()),
            "n_init_cluster":   int(self.n_init_cluster.get()),
            "cluster_id_col":   self.cluster_id_col.get().strip() or "cluster_id",
            "cluster_kw_col":   self.cluster_kw_col.get().strip() or "cluster_keywords",
            "use_sbert_cluster": bool(self.use_sbert_cluster.get()),
            # — роли в кластеризации (не зависят от вкладки Обучение) —
            "ignore_chatbot_cluster": bool(self.ignore_chatbot_cluster.get()),
            "cluster_role_mode":      self.cluster_role_mode.get() or "all",
            # — расширенные методы кластеризации —
            "use_combo_cluster":        bool(self.use_combo_cluster.get()),
            "combo_svd_dim":            max(50, int(self.combo_svd_dim.get())),
            "combo_alpha":              max(0.0, min(1.0, float(self.combo_alpha.get()))),
            "k_score_method":           self.k_score_method.get() or "elbow",
            "use_cosine_cluster":       bool(self.use_cosine_cluster.get()),
            "use_umap":                 bool(self.use_umap.get()),
            "umap_n_components":        max(5, int(self.umap_n_components.get())),
            "umap_n_neighbors":         max(2, int(self.umap_n_neighbors.get())),
            "umap_min_dist":            max(0.0, min(1.0, float(self.umap_min_dist.get()))),
            "umap_metric":              self.umap_metric.get().strip() or "cosine",
            "use_hdbscan":              bool(self.use_hdbscan.get()),
            # 0 пропускаем без клампа — он означает «auto» в worker'е
            # (app_cluster._cluster_step_cluster). Любое положительное → ≥2.
            "hdbscan_min_cluster_size": (
                0 if int(self.hdbscan_min_cluster_size.get()) <= 0
                else max(2, int(self.hdbscan_min_cluster_size.get()))
            ),
            "hdbscan_min_samples":     max(0, int(self.hdbscan_min_samples.get())),
            "hdbscan_eps":             max(0.0, float(self.hdbscan_eps.get())),
            # — новые селекторы режима кластеризации —
            "cluster_vec_mode":  self.cluster_vec_mode.get() or "tfidf",
            "cluster_algo":      self.cluster_algo.get() or "kmeans",
            # — LDA —
            "lda_n_topics":      max(2, int(self.lda_n_topics.get())),
            "lda_max_iter":      max(10, int(self.lda_max_iter.get())),
            "lda_topics_col":    self.lda_topics_col.get().strip() or "lda_topics",
            # — Иерархическая кластеризация —
            "hier_k_top":        max(2, int(self.hier_k_top.get())),
            "hier_k_sub":        max(2, int(self.hier_k_sub.get())),
            "hier_min_sub":      max(5, int(self.hier_min_sub.get())),
            "hier_l1_col":       self.hier_l1_col.get().strip() or "cluster_l1",
            # — BERTopic —
            "bertopic_min_topic_size": max(2, int(self.bertopic_min_topic_size.get())),
            "bertopic_nr_topics":      self.bertopic_nr_topics.get().strip() or "auto",
            # — Предобработка текста для кластеризации —
            "use_lemma_cluster":  bool(self.use_lemma_cluster.get()),
            "normalize_numbers":  bool(self.normalize_numbers.get()),
            # — SVD для TF-IDF кластеризации —
            "use_tfidf_svd":      bool(self.use_tfidf_svd.get()),
            "tfidf_svd_dim":      max(20, int(self.tfidf_svd_dim.get())),
            # — c-TF-IDF ключевые слова —
            "use_ctfidf_keywords": bool(self.use_ctfidf_keywords.get()),
            # — Seed воспроизводимости —
            "cluster_random_seed": max(0, int(self.cluster_random_seed.get())),
            # — Семантические якоря —
            "use_anchors":       bool(self.use_anchors.get()),
            "anchor_phrases":    [],  # заполняется в run_cluster() до запуска потока
            # — T5-суммаризация —
            "use_t5_summary":   bool(self.use_t5_summary.get()),
            "t5_summary_col":   self.t5_summary_col.get().strip() or "t5_summary",
            "t5_model_name":    self.t5_model_name.get().strip() or "UrukHan/t5-russian-summarization",
            "t5_max_input":     max(64, int(self.t5_max_input.get())),
            "t5_max_output":    max(32, int(self.t5_max_output.get())),
            "t5_batch_size":    max(1, int(self.t5_batch_size.get())),
            # — LLM-нейминг кластеров —
            "use_llm_naming":   bool(self.use_llm_naming.get()),
            "llm_api_key":      self.llm_api_key.get().strip(),
            "llm_provider":     self.llm_provider.get().strip() or "anthropic",
            "llm_model":        self.llm_model.get().strip() or "claude-sonnet-4-6",
            "llm_name_col":     self.llm_name_col.get().strip() or "cluster_name",
            "use_llm_reason_summary": bool(self.use_llm_reason_summary.get()),
            "llm_reason_col":   self.llm_reason_col.get().strip() or "cluster_reason",
            "use_rule_reason_summary": bool(self.use_rule_reason_summary.get()),
            "merge_similar_clusters": bool(self.merge_similar_clusters.get()),
            "merge_threshold":        max(0.5, min(0.99, float(self.merge_threshold.get()))),
            "n_repr_examples":        max(3, min(10, int(self.n_repr_examples.get()))),
            # — PCA перед UMAP —
            "use_pca_before_umap": bool(self.use_pca_before_umap.get()),
            "pca_n_components":    max(10, int(self.pca_n_components.get())),
            # — FASTopic —
            "fastopic_n_top_words": max(5, int(self.fastopic_n_top_words.get())),
            "fastopic_theta":       max(0.01, min(1.0, float(self.fastopic_theta.get()))),
            # — Качество кластеров —
            "show_cluster_quality": bool(self.show_cluster_quality.get()),
            "cluster_quality_col":  self.cluster_quality_col.get().strip() or "cluster_quality",
            # — Второй KMeans по выбросам HDBSCAN —
            "recluster_noise":   bool(self.recluster_noise.get()),
            "noise_k_clusters":  max(2, int(self.noise_k_clusters.get())),
            "noise_cluster_col": self.noise_cluster_col.get().strip() or "noise_cluster_id",
            # — min_df для TF-IDF —
            "cluster_min_df":    max(0, int(self.cluster_min_df.get())),
            # — Семантическая дедупликация —
            "use_dedup":         bool(self.use_dedup.get()),
            "dedup_threshold":   max(0.5, min(1.0, float(self.dedup_threshold.get()))),
            # — Интерактивная визуализация —
            "use_cluster_viz":   bool(self.use_cluster_viz.get()),
            # — Инкрементальная кластеризация (#12) —
            "save_cluster_model":      bool(self.save_cluster_model.get()),
            "cluster_model_path":      self.cluster_model_path.get().strip(),
            "use_saved_cluster_model": bool(self.use_saved_cluster_model.get()),
            "diagnostic_mode":         bool(self.diagnostic_mode.get()),
            # — Потоковый режим (#20) —
            "use_streaming_cluster":   bool(self.use_streaming_cluster.get()),
            "streaming_chunk_size":    max(500, int(self.streaming_chunk_size.get())),
            # — ClusterLLM feedback loop (#13) —
            "use_llm_feedback":        bool(self.use_llm_feedback.get()),
            "llm_feedback_col":        self.llm_feedback_col.get().strip() or "llm_feedback",
        }
