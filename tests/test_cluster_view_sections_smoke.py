import tkinter as tk
from tkinter import ttk

import pytest

from app_cluster_view import (
    build_cluster_algo_main_section,
    build_cluster_basic_settings_sections,
    build_cluster_files_card,
)


class _DummyApp:
    def __init__(self, root):
        self._root = root
        self.desc_col = tk.StringVar(value="desc")
        self.call_col = tk.StringVar(value="call")
        self.chat_col = tk.StringVar(value="chat")
        self.k_clusters = tk.IntVar(value=10)
        self.use_elbow = tk.BooleanVar(value=False)
        self.ignore_chatbot_cluster = tk.BooleanVar(value=True)
        self.cluster_role_mode = tk.StringVar(value="all")
        self.use_stop_words = tk.BooleanVar(value=True)
        self.use_noise_tokens = tk.BooleanVar(value=True)
        self.use_noise_phrases = tk.BooleanVar(value=True)
        self.cluster_vec_mode = tk.StringVar(value="tfidf")
        self.use_ctfidf_keywords = tk.BooleanVar(value=True)
        self.cluster_algo = tk.StringVar(value="kmeans")

    def _auto_detect_cluster_params(self):
        return None

    def _combobox(self, parent, r, label, var, _help):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var).grid(row=r, column=1, sticky="we")

    def attach_help(self, *_args, **_kwargs):
        return None

    def add_cluster_files(self):
        return None

    def add_cluster_folder(self):
        return None

    def remove_cluster_file(self):
        return None

    def clear_cluster_files(self):
        return None

    def _sync_cluster_file_buttons(self):
        return None


def test_cluster_basic_sections_smoke():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("tk display unavailable in test environment")
    root.withdraw()
    try:
        app = _DummyApp(root)
        parent = ttk.Frame(root)
        parent.pack()
        card = build_cluster_files_card(app, parent)
        build_cluster_basic_settings_sections(app, parent, card)
        build_cluster_algo_main_section(app, parent)
        assert hasattr(app, "_cluster_k_widgets")
        assert len(app._cluster_k_widgets) == 2
        assert hasattr(app, "_detect_status_var")
        assert hasattr(app, "_cluster_algo_rbs")
        assert len(app._cluster_algo_rbs) >= 4
    finally:
        root.destroy()
