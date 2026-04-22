# -*- coding: utf-8 -*-
"""
Unit tests for LLMClient._parse_retry_after helper.

Covers RFC 7231 §7.1.3 Retry-After header parsing:
  * delta-seconds (integer)
  * HTTP-date (RFC 1123 / IMF-fixdate)
  * negative / empty / missing / malformed inputs
  * case-insensitive header lookup
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llm_client import LLMClient


class TestParseRetryAfter:
    def test_none_headers_returns_none(self):
        assert LLMClient._parse_retry_after(None) is None

    def test_missing_header_returns_none(self):
        assert LLMClient._parse_retry_after({}) is None

    def test_empty_value_returns_none(self):
        assert LLMClient._parse_retry_after({"Retry-After": ""}) is None

    def test_integer_delta_seconds(self):
        assert LLMClient._parse_retry_after({"Retry-After": "5"}) == 5.0

    def test_float_delta_seconds(self):
        assert LLMClient._parse_retry_after({"Retry-After": "2.5"}) == 2.5

    def test_zero_delta(self):
        assert LLMClient._parse_retry_after({"Retry-After": "0"}) == 0.0

    def test_negative_clamped_to_zero(self):
        assert LLMClient._parse_retry_after({"Retry-After": "-3"}) == 0.0

    def test_lowercase_header_accepted(self):
        assert LLMClient._parse_retry_after({"retry-after": "7"}) == 7.0

    def test_malformed_string_returns_none(self):
        assert LLMClient._parse_retry_after({"Retry-After": "soon"}) is None

    def test_http_date_future(self):
        future = datetime.now(timezone.utc) + timedelta(seconds=30)
        # IMF-fixdate format
        date_str = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        secs = LLMClient._parse_retry_after({"Retry-After": date_str})
        assert secs is not None
        # Allow ±2s slack for test timing
        assert 28.0 <= secs <= 32.0

    def test_http_date_past_clamped_to_zero(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=60)
        date_str = past.strftime("%a, %d %b %Y %H:%M:%S GMT")
        assert LLMClient._parse_retry_after({"Retry-After": date_str}) == 0.0

    def test_headers_without_get_method_returns_none(self):
        class _Bad:
            pass

        assert LLMClient._parse_retry_after(_Bad()) is None
