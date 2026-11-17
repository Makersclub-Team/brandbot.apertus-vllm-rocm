# SPDX-License-Identifier: Apache-2.0
"""Offline protocol checks only: these do not establish GPU or model compatibility."""

import importlib.util
import io
import json
import time
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "smoke.py"
SPEC = importlib.util.spec_from_file_location("apertus_smoke", SOURCE)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def event(value):
    payload = value if isinstance(value, str) else json.dumps(value)
    return ("data: " + payload + "\r\n\r\n").encode("utf-8")


def chunk(content=None, finish_reason=None, **delta):
    if content is not None:
        delta["content"] = content
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def response(content="Hello", finish_reason="stop", **message):
    return {"choices": [{"message": {"role": "assistant", "content": content, **message},
                         "finish_reason": finish_reason}]}


class SmokeProtocolTests(unittest.TestCase):
    def validate(self, data):
        return smoke.validate_text_stream(io.BytesIO(data), time.monotonic() + 5)

    def test_sse_crlf_comments_and_multiline_fields(self):
        data = b': keepalive\r\ndata: {"choices":\r\ndata: []}\r\n\r\ndata: [DONE]\r\n\r\n'
        payloads = list(smoke.sse_payloads(io.BytesIO(data), time.monotonic() + 5))
        self.assertEqual(json.loads(payloads[0]), {"choices": []})
        self.assertEqual(payloads[1], "[DONE]")

    def test_completed_text_stream(self):
        data = (b": keepalive\r\n\r\n" + event(chunk("Hello")) + event(chunk("!", "stop"))
                + event({"choices": [], "usage": {"completion_tokens": 2}}) + event("[DONE]"))
        result = self.validate(data)
        self.assertTrue(result["done_marker_received"])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["visible_content_characters"], 6)

    def test_missing_done_marker_fails(self):
        with self.assertRaisesRegex(ValueError, "DONE"):
            self.validate(event(chunk("Hello", "stop")))

    def test_missing_successful_finish_reason_fails(self):
        for reason in (None, "length", "content_filter", "tool_calls"):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, "finish"):
                self.validate(event(chunk("Hello", reason)) + event("[DONE]"))

    def test_reasoning_without_visible_text_does_not_pass(self):
        data = event(chunk(finish_reason="stop", reasoning="Internal reasoning")) + event("[DONE]")
        with self.assertRaisesRegex(ValueError, "no visible text"):
            self.validate(data)

    def test_malformed_sse_json_fails(self):
        with self.assertRaises(json.JSONDecodeError):
            self.validate(event("{not-json}") + event("[DONE]"))

    def test_sse_api_error_fails(self):
        with self.assertRaisesRegex(ValueError, "error"):
            self.validate(event({"error": {"message": "example error"}}) + event("[DONE]"))

    def test_expired_stream_budget_fails(self):
        with self.assertRaises(TimeoutError):
            smoke.validate_text_stream(io.BytesIO(event("[DONE]")), time.monotonic() - 1)

    def test_final_text(self):
        self.assertEqual(smoke.final_text(response()), ("Hello", "stop"))

    def test_invalid_nonstream_responses_fail(self):
        cases = [None, {}, {"choices": []}, {"error": {"message": "example error"}},
                 response(""), response("   "), response(None), response(finish_reason="length"),
                 response(finish_reason=None), response(role="user"), response(tool_calls=[{"id": "test"}])]
        for value in cases:
            with self.subTest(response=value), self.assertRaises(ValueError):
                smoke.final_text(value)

    def test_expected_numeric_result(self):
        for content in ("42", "42.", "The result is 42.", "42.0", "The result is 42.00."):
            with self.subTest(content=content):
                self.assertTrue(smoke.has_result_42(content))
        for content in ("-42", "−42", "142", "42.5", "42.01", "42,5", "42e2", "value42", "forty-two"):
            with self.subTest(content=content):
                self.assertFalse(smoke.has_result_42(content))


if __name__ == "__main__":
    unittest.main()
