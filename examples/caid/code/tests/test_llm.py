"""Tests for caid.llm module."""

from __future__ import annotations

import pytest

from caid.config import LLMConfig
from caid.llm import LLMClient, extract_json


class TestExtractJson:
    def test_json_code_block(self) -> None:
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        assert extract_json(text) == '{"key": "value"}'

    def test_plain_code_block(self) -> None:
        text = 'Result:\n```\n{"a": 1}\n```'
        assert extract_json(text) == '{"a": 1}'

    def test_raw_json(self) -> None:
        assert extract_json('{"direct": true}') == '{"direct": true}'

    def test_embedded_json(self) -> None:
        text = 'Some prefix {"embedded": 1} suffix'
        assert extract_json(text) == '{"embedded": 1}'

    def test_array_json(self) -> None:
        assert extract_json('[1, 2, 3]') == '[1, 2, 3]'

    def test_nested_json(self) -> None:
        text = '```json\n{"a": {"b": [1, 2]}}\n```'
        assert extract_json(text) == '{"a": {"b": [1, 2]}}'

    def test_multiline_json(self) -> None:
        text = '```json\n{\n  "key": "value",\n  "num": 42\n}\n```'
        result = extract_json(text)
        assert '"key"' in result
        assert '"num"' in result


class TestLLMClient:
    def test_initialization(self) -> None:
        config = LLMConfig(model="test", api_key="key")
        client = LLMClient(config)
        assert client.total_prompt_tokens == 0
        assert client.total_completion_tokens == 0

    def test_token_usage_property(self) -> None:
        config = LLMConfig(model="test")
        client = LLMClient(config)
        usage = client.token_usage
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_manual_token_tracking(self) -> None:
        config = LLMConfig(model="test")
        client = LLMClient(config)
        client.total_prompt_tokens = 100
        client.total_completion_tokens = 50
        usage = client.token_usage
        assert usage["total_tokens"] == 150
