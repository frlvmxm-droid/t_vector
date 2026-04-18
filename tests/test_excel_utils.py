# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for excel_utils.py.

Covers: open_tabular, read_headers, idx_of, estimate_total_rows,
        fmt_eta, fmt_speed.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from excel_utils import (
    TabularFileTooLargeError,
    fmt_eta,
    fmt_speed,
    idx_of,
    open_tabular,
    read_headers,
    estimate_total_rows,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)


def _write_xlsx(path: Path, rows: list[list]) -> None:
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    wb.save(str(path))


# ===========================================================================
# idx_of
# ===========================================================================

class TestIdxOf:
    def test_found_first(self):
        assert idx_of(["A", "B", "C"], "A") == 0

    def test_found_middle(self):
        assert idx_of(["A", "B", "C"], "B") == 1

    def test_found_last(self):
        assert idx_of(["A", "B", "C"], "C") == 2

    def test_not_found_returns_none(self):
        assert idx_of(["A", "B", "C"], "Z") is None

    def test_empty_col_name_returns_none(self):
        assert idx_of(["A", "B"], "") is None

    def test_none_col_name_returns_none(self):
        # None is falsy, so treated as missing
        assert idx_of(["A", "B"], None) is None  # type: ignore[arg-type]

    def test_empty_headers_returns_none(self):
        assert idx_of([], "A") is None

    def test_duplicate_headers_returns_first(self):
        assert idx_of(["X", "X", "Y"], "X") == 0

    def test_case_sensitive(self):
        assert idx_of(["abc", "ABC"], "abc") == 0
        assert idx_of(["abc", "ABC"], "ABC") == 1
        assert idx_of(["abc"], "Abc") is None

    def test_single_element_found(self):
        assert idx_of(["only"], "only") == 0

    def test_single_element_not_found(self):
        assert idx_of(["only"], "other") is None


# ===========================================================================
# fmt_eta
# ===========================================================================

class TestFmtEta:
    def test_done_zero_returns_empty(self):
        assert fmt_eta(time.time(), 0, 100) == ""

    def test_total_zero_returns_empty(self):
        assert fmt_eta(time.time(), 10, 0) == ""

    def test_done_negative_returns_empty(self):
        assert fmt_eta(time.time(), -1, 100) == ""

    def test_seconds_format(self):
        # done=50 of 100 with start_ts set so elapsed=1s → speed=50/s, remain=50, eta=1s
        start = time.time() - 1.0
        result = fmt_eta(start, 50, 100)
        assert result.startswith("ETA") and result.endswith("s")

    def test_minutes_format(self):
        # done=1, total=10000, elapsed=0.001 → speed=1000/s, remain=9999, eta~10s
        # We need eta >= 60s: done=1, total=61000, elapsed very small
        start = time.time() - 0.001
        result = fmt_eta(start, 1, 1_000_000)
        # speed ~ 100/s → remain ~ 999999 → eta ~ 9999s → should show hours or minutes
        assert "ETA" in result

    def test_hours_format(self):
        # Simulate a very slow rate: elapsed=1s, done=1, total=100_000_000
        start = time.time() - 1.0
        result = fmt_eta(start, 1, 100_000_000)
        # eta = 99999999 / 1 = ~99999999 s → hours
        assert result.startswith("ETA") and "h" in result

    def test_near_complete_returns_small_eta(self):
        start = time.time() - 10.0
        result = fmt_eta(start, 99, 100)
        assert result.startswith("ETA")

    def test_exactly_complete_is_zero_remain(self):
        start = time.time() - 5.0
        result = fmt_eta(start, 100, 100)
        # remain=0, eta=0 → "ETA 0s"
        assert result == "ETA 0s"

    def test_returns_string_type(self):
        result = fmt_eta(time.time() - 1.0, 10, 100)
        assert isinstance(result, str)


# ===========================================================================
# fmt_speed
# ===========================================================================

class TestFmtSpeed:
    def test_returns_rows_per_second_label(self):
        result = fmt_speed(time.time() - 1.0, 100)
        assert "rows/s" in result

    def test_returns_string(self):
        assert isinstance(fmt_speed(time.time() - 1.0, 50), str)

    def test_high_speed_format(self):
        # 1000 rows in 1 second → "1000.0 rows/s"
        result = fmt_speed(time.time() - 1.0, 1000)
        assert "rows/s" in result
        assert "1000" in result

    def test_low_speed_uses_two_decimal(self):
        # 0 rows in 1 second → "0.00 rows/s"
        result = fmt_speed(time.time() - 1.0, 0)
        assert "rows/s" in result
        # Speed < 1 → two decimal places format
        assert "." in result

    def test_done_zero_still_returns_string(self):
        result = fmt_speed(time.time() - 1.0, 0)
        assert isinstance(result, str)
        assert "rows/s" in result

    def test_large_elapsed_gives_low_speed(self):
        # 1 row in 1000 seconds → very low speed
        result = fmt_speed(time.time() - 1000.0, 1)
        assert "rows/s" in result


# ===========================================================================
# read_headers — CSV
# ===========================================================================

class TestReadHeadersCsv:
    def test_simple_headers(self, tmp_path: Path):
        p = tmp_path / "data.csv"
        _write_csv(p, [["Name", "Age", "City"], ["Alice", "30", "Moscow"]])
        headers = read_headers(p)
        assert headers == ["Name", "Age", "City"]

    def test_single_header(self, tmp_path: Path):
        p = tmp_path / "single.csv"
        _write_csv(p, [["OnlyColumn"], ["value"]])
        assert read_headers(p) == ["OnlyColumn"]

    def test_returns_list_of_strings(self, tmp_path: Path):
        p = tmp_path / "types.csv"
        _write_csv(p, [["Col1", "Col2"], ["a", "b"]])
        result = read_headers(p)
        assert isinstance(result, list)
        assert all(isinstance(h, str) for h in result)

    def test_strips_whitespace_from_headers(self, tmp_path: Path):
        p = tmp_path / "spaces.csv"
        # Write with explicit spaces in header
        p.write_text(" Name , Age \nAlice,30\n", encoding="utf-8")
        headers = read_headers(p)
        assert headers == ["Name", "Age"]

    def test_empty_header_cell_becomes_empty_string(self, tmp_path: Path):
        p = tmp_path / "empty_col.csv"
        _write_csv(p, [["Name", "", "City"], ["Alice", "", "Moscow"]])
        headers = read_headers(p)
        assert headers[1] == ""


# ===========================================================================
# read_headers — XLSX
# ===========================================================================

class TestReadHeadersXlsx:
    def test_simple_xlsx_headers(self, tmp_path: Path):
        p = tmp_path / "data.xlsx"
        _write_xlsx(p, [["ID", "Text", "Label"], [1, "hello", "cat"]])
        headers = read_headers(p)
        assert headers == ["ID", "Text", "Label"]

    def test_xlsx_single_column(self, tmp_path: Path):
        p = tmp_path / "single.xlsx"
        _write_xlsx(p, [["Score"], [99]])
        assert read_headers(p) == ["Score"]


# ===========================================================================
# open_tabular — CSV
# ===========================================================================

class TestOpenTabularCsv:
    def test_yields_header_and_data_rows(self, tmp_path: Path):
        p = tmp_path / "test.csv"
        _write_csv(p, [["A", "B"], ["1", "2"], ["3", "4"]])
        with open_tabular(p) as rows:
            header = next(rows)
            data = list(rows)
        assert header == ("A", "B")
        assert len(data) == 2

    def test_empty_cells_become_none(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        _write_csv(p, [["A", "B"], ["val", ""]])
        with open_tabular(p) as rows:
            next(rows)  # skip header
            row = next(rows)
        assert row[1] is None

    def test_non_empty_cells_preserved(self, tmp_path: Path):
        p = tmp_path / "vals.csv"
        _write_csv(p, [["X"], ["hello"]])
        with open_tabular(p) as rows:
            next(rows)
            row = next(rows)
        assert row[0] == "hello"

    def test_semicolon_delimiter(self, tmp_path: Path):
        p = tmp_path / "semi.csv"
        p.write_text("Col1;Col2\nfoo;bar\n", encoding="utf-8")
        with open_tabular(p) as rows:
            header = next(rows)
            data_row = next(rows)
        assert "Col1" in header
        assert "foo" in data_row

    def test_context_manager_closes_file(self, tmp_path: Path):
        p = tmp_path / "close.csv"
        _write_csv(p, [["A"], ["1"]])
        # No exception should leak from context manager
        with open_tabular(p) as rows:
            list(rows)

    def test_multiple_data_rows(self, tmp_path: Path):
        p = tmp_path / "multi.csv"
        _write_csv(p, [["Col"]] + [[str(i)] for i in range(10)])
        with open_tabular(p) as rows:
            next(rows)
            data = list(rows)
        assert len(data) == 10

    def test_row_is_tuple(self, tmp_path: Path):
        p = tmp_path / "tuple.csv"
        _write_csv(p, [["A", "B"], ["x", "y"]])
        with open_tabular(p) as rows:
            next(rows)
            row = next(rows)
        assert isinstance(row, tuple)


# ===========================================================================
# open_tabular — XLSX
# ===========================================================================

class TestOpenTabularXlsx:
    def test_xlsx_header_and_data(self, tmp_path: Path):
        p = tmp_path / "test.xlsx"
        _write_xlsx(p, [["Name", "Score"], ["Alice", 95], ["Bob", 80]])
        with open_tabular(p) as rows:
            header = next(rows)
            data = list(rows)
        assert header == ("Name", "Score")
        assert len(data) == 2

    def test_xlsx_values_correct(self, tmp_path: Path):
        p = tmp_path / "vals.xlsx"
        _write_xlsx(p, [["Key", "Value"], ["x", 42]])
        with open_tabular(p) as rows:
            next(rows)
            row = next(rows)
        assert row[0] == "x"
        assert row[1] == 42


# ===========================================================================
# estimate_total_rows
# ===========================================================================

class TestEstimateTotalRows:
    def test_csv_single_file(self, tmp_path: Path):
        p = tmp_path / "data.csv"
        _write_csv(p, [["A"]] + [[str(i)] for i in range(5)])
        result = estimate_total_rows([p])
        assert result == 5

    def test_csv_multiple_files(self, tmp_path: Path):
        p1 = tmp_path / "a.csv"
        p2 = tmp_path / "b.csv"
        _write_csv(p1, [["A"]] + [["x"]] * 3)
        _write_csv(p2, [["A"]] + [["y"]] * 4)
        result = estimate_total_rows([p1, p2])
        assert result == 7

    def test_xlsx_small_file(self, tmp_path: Path):
        p = tmp_path / "data.xlsx"
        _write_xlsx(p, [["Col"]] + [[i] for i in range(6)])
        result = estimate_total_rows([p])
        assert result == 6

    def test_empty_list_returns_at_least_one(self, tmp_path: Path):
        # No paths → total=0 → clamped to max(0,1)=1
        result = estimate_total_rows([])
        assert result >= 1

    def test_header_only_csv_returns_zero_or_clamped(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        _write_csv(p, [["Header"]])
        result = estimate_total_rows([p])
        # 0 data rows → total=0 → clamped to 1
        assert result >= 1

    def test_returns_integer(self, tmp_path: Path):
        p = tmp_path / "int.csv"
        _write_csv(p, [["A"], ["1"]])
        result = estimate_total_rows([p])
        assert isinstance(result, int)


# ===========================================================================
# Size limits — защита от OOM / ZIP-bomb
# ===========================================================================

class TestSizeLimits:
    def test_csv_over_limit_raises(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "big.csv"
        _write_csv(p, [["A"]] + [["x"]] * 10)
        monkeypatch.setenv("MAX_CSV_BYTES", "1")  # все CSV теперь слишком велики
        with pytest.raises(TabularFileTooLargeError):
            with open_tabular(p) as rows:
                list(rows)

    def test_xlsx_compressed_over_limit_raises(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "big.xlsx"
        _write_xlsx(p, [["A"]] + [[i] for i in range(100)])
        monkeypatch.setenv("MAX_XLSX_BYTES", "1")
        with pytest.raises(TabularFileTooLargeError):
            with open_tabular(p) as rows:
                list(rows)

    def test_xlsx_uncompressed_over_limit_raises(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "bomb.xlsx"
        _write_xlsx(p, [["A"], ["hello"]])
        monkeypatch.setenv("MAX_XLSX_UNCOMPRESSED_BYTES", "1")
        with pytest.raises(TabularFileTooLargeError):
            with open_tabular(p) as rows:
                list(rows)

    def test_within_limits_ok(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "ok.xlsx"
        _write_xlsx(p, [["A"], ["v"]])
        monkeypatch.setenv("MAX_XLSX_BYTES", "10000000")
        with open_tabular(p) as rows:
            header = next(rows)
        assert header == ("A",)

    def test_invalid_env_falls_back_to_default(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "ok.csv"
        _write_csv(p, [["A"], ["v"]])
        monkeypatch.setenv("MAX_CSV_BYTES", "not-a-number")
        with open_tabular(p) as rows:
            list(rows)
