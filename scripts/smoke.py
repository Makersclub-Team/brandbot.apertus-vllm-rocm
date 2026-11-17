#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check text and optional tool calls against an existing OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator


MAX_BODY_BYTES = 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sse_payloads(stream: BinaryIO, deadline: float) -> Iterator[str]:
    """Read SSE data fields, accepting CRLF, comments, and multi-line events."""
    data: list[str] = []
    total_bytes = 0
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("The stream did not complete within the request time budget.")
        raw = stream.readline(MAX_BODY_BYTES + 1)
        if not raw:
            if data:
                yield "\n".join(data)
            return
        total_bytes += len(raw)
        require(len(raw) <= MAX_BODY_BYTES and total_bytes <= 16 * MAX_BODY_BYTES,
                "The SSE response exceeded the smoke-test size limit.")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif not line.startswith(":"):
            field, separator, value = line.partition(":")
            if field == "data":
                data.append(value[1:] if separator and value.startswith(" ") else value)


def first_choice(response: Any) -> dict[str, Any]:
    require(isinstance(response, dict), "The API response is not a JSON object.")
    require("error" not in response, "The API returned an error object.")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "Expected exactly one response choice.")
    choice = choices[0]
    require(isinstance(choice, dict), "The response choice is not an object.")
    return choice


def final_text(response: Any) -> tuple[str, str]:
    choice = first_choice(response)
    message = choice.get("message")
    require(isinstance(message, dict) and message.get("role") == "assistant",
            "Expected an assistant message.")
    content = message.get("content")
    require(isinstance(content, str) and bool(content.strip()), "The assistant returned no visible text.")
    require(not message.get("tool_calls"), "Unexpected tool calls in a text response.")
    require(choice.get("finish_reason") == "stop", "The text response did not finish with reason 'stop'.")
    return content, choice["finish_reason"]


def validate_text_stream(stream: BinaryIO, deadline: float) -> dict[str, Any]:
    """Require visible text, a successful finish reason, and the SSE terminator."""
    fragments: list[str] = []
    finish_reason = None
    done = False
    chunks = 0
    for payload in sse_payloads(stream, deadline):
        if payload.strip() == "[DONE]":
            done = True
            break
        chunk = json.loads(payload)
        require(isinstance(chunk, dict) and "error" not in chunk, "The stream returned an error or invalid object.")
        choices = chunk.get("choices")
        require(isinstance(choices, list), "A streaming chunk has no choices array.")
        if not choices:  # Optional final usage-only chunk.
            continue
        require(len(choices) == 1 and isinstance(choices[0], dict), "Expected one streaming choice.")
        choice = choices[0]
        require(choice.get("index", 0) == 0, "Unexpected streaming choice index.")
        delta = choice.get("delta")
        require(isinstance(delta, dict), "A streaming choice has no delta object.")
        require(not delta.get("tool_calls"), "Unexpected tool calls in the greeting stream.")
        content = delta.get("content")
        if content is not None:
            require(isinstance(content, str), "The streamed content is not a string.")
            fragments.append(content)
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
        chunks += 1
    visible = "".join(fragments)
    require(bool(visible.strip()), "The completed stream contained no visible text.")
    require(finish_reason == "stop", "The stream did not finish with reason 'stop'.")
    require(done, "The stream ended without the [DONE] marker.")
    return {"visible_content_characters": len(visible), "finish_reason": finish_reason,
            "done_marker_received": done, "choice_chunks": chunks}


def has_result_42(content: str) -> bool:
    """Accept the numeric result, including a sentence-final period, but not -42 or 142."""
    return bool(re.search(r"(?<![\w.+\-\u2212])42(?:\.0+)?(?!\w|[.,]\d)", content))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact served model name.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--output", type=Path, default=Path("smoke-report.json"))
    parser.add_argument("--tools", action="store_true", help="Also check an automatic tool call and its result roundtrip.")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Socket timeout and SSE request time budget in seconds (default: 60).")
    args = parser.parse_args()
    require(math.isfinite(args.timeout) and args.timeout > 0, "--timeout must be a positive finite number.")
    parsed = urllib.parse.urlsplit(args.base_url)
    require(parsed.scheme in ("http", "https") and bool(parsed.netloc), "--base-url must be an HTTP(S) URL.")
    require(not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
            "--base-url must not contain credentials, query parameters, or a fragment.")
    base_url = args.base_url.rstrip("/")
    api_key = os.environ.get("VLLM_API_KEY", "")

    report: dict[str, Any] = {
        "schema_version": 1, "kind": "apertus_text_api_smoke",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url, "model": args.model,
        "scope": "Text API and optional non-streaming tool roundtrip only; no performance measurements.",
        "checks": [],
        "not_run": ["image_input", "audio_input", "video_input", "long_context",
                    "concurrent_requests", "performance", "streaming_tool_calls", "thinking_with_tools"],
    }

    def safe_error(exc: Exception) -> str:
        text = str(exc)
        if api_key:
            text = text.replace(api_key, "[REDACTED]")
        return text[:1000]

    def run(name: str, operation: Callable[[], dict[str, Any]]) -> bool:
        try:
            details = operation()
        except Exception as exc:
            report["checks"].append({"name": name, "status": "failed",
                                     "error_type": type(exc).__name__, "error": safe_error(exc)})
            return False
        report["checks"].append({"name": name, "status": "passed", "details": details})
        return True

    def request(payload: dict[str, Any]) -> Any:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        req = urllib.request.Request(base_url + "/chat/completions", method="POST",
                                     data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=args.timeout)
        except urllib.error.HTTPError as exc:
            # Do not log headers, credentials, or arbitrary server error bodies.
            exc.close()
            raise RuntimeError(f"The server returned HTTP {exc.code}; inspect its logs for details.") from None

    def json_request(payload: dict[str, Any]) -> Any:
        with request(payload) as response:
            body = response.read(MAX_BODY_BYTES + 1)
        require(len(body) <= MAX_BODY_BYTES, "The JSON response exceeded the smoke-test size limit.")
        return json.loads(body)

    def body(messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        return {"model": args.model, "messages": messages, "temperature": 0,
                "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}, **extra}

    greeting = [{"role": "user", "content": "Reply with one short greeting."}]

    def check_nonstream() -> dict[str, Any]:
        content, reason = final_text(json_request(body(greeting, stream=False)))
        return {"visible_content_characters": len(content), "finish_reason": reason,
                "thinking_requested": False}

    run("chat.non_streaming", check_nonstream)

    def check_stream() -> dict[str, Any]:
        deadline = time.monotonic() + args.timeout
        with request(body(greeting, stream=True)) as response:
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            require(content_type == "text/event-stream", "Expected a text/event-stream response.")
            result = validate_text_stream(response, deadline)
        return {**result, "thinking_requested": False}

    run("chat.streaming_sse", check_stream)

    if args.tools:
        tools = [{"type": "function", "function": {
            "name": "add_numbers", "description": "Add the two supplied numbers.",
            "parameters": {"type": "object", "properties": {"a": {"type": "number"},
                           "b": {"type": "number"}}, "required": ["a", "b"], "additionalProperties": False}}}]
        messages = [{"role": "user", "content": "Use the add_numbers tool to compute 17 + 25. "
                     "You must call the tool. After receiving the result, answer with its numeric value."}]
        tool_state: dict[str, Any] = {}

        def check_auto_tool() -> dict[str, Any]:
            choice = first_choice(json_request(body(messages, tools=tools, tool_choice="auto",
                                                   max_tokens=256, stream=False)))
            message = choice.get("message")
            require(isinstance(message, dict) and message.get("role") == "assistant", "Expected an assistant tool-call message.")
            calls = message.get("tool_calls")
            require(isinstance(calls, list) and len(calls) == 1, "Expected exactly one automatic tool call.")
            call = calls[0]
            require(isinstance(call, dict) and call.get("type") == "function", "Expected a function tool call.")
            require(isinstance(call.get("id"), str) and bool(call["id"]), "The tool call has no ID.")
            function = call.get("function")
            require(isinstance(function, dict) and function.get("name") == "add_numbers", "The wrong tool was selected.")
            require(isinstance(function.get("arguments"), str), "Tool arguments must be a JSON string.")
            arguments = json.loads(function["arguments"])
            require(isinstance(arguments, dict) and set(arguments) == {"a", "b"}, "Tool arguments have an unexpected shape.")
            values = [arguments["a"], arguments["b"]]
            require(all(type(value) in (int, float) and math.isfinite(value) for value in values), "Tool arguments must be finite numbers.")
            require(sorted(values) == [17, 25], "The tool was not called with the requested operands 17 and 25.")
            require(choice.get("finish_reason") == "tool_calls", "The tool response did not finish with reason 'tool_calls'.")
            result = sum(values)
            tool_state.update({"call": call, "message": {"role": "assistant", "content": message.get("content"),
                                                        "tool_calls": calls}, "result": result})
            return {"tool_name": "add_numbers", "operands": values, "calculated_result": result,
                    "tool_choice": "auto", "finish_reason": choice["finish_reason"]}

        if run("tools.auto_call", check_auto_tool):
            def check_tool_roundtrip() -> dict[str, Any]:
                followup = messages + [tool_state["message"], {"role": "tool", "tool_call_id": tool_state["call"]["id"],
                                      "content": json.dumps({"result": tool_state["result"]})}]
                content, reason = final_text(json_request(body(followup, tools=tools, tool_choice="none", stream=False)))
                require(has_result_42(content),
                        "The final answer did not contain the expected numeric result 42.")
                return {"expected_result": 42, "numeric_result_present": True, "finish_reason": reason,
                        "visible_content_characters": len(content)}

            run("tools.result_roundtrip", check_tool_roundtrip)
        else:
            report["checks"].append({"name": "tools.result_roundtrip", "status": "not_run",
                                     "reason": "The automatic tool-call check failed."})
    else:
        report["not_run"].extend(["tools.auto_call", "tools.result_roundtrip"])

    report["status"] = "failed" if any(c["status"] == "failed" for c in report["checks"]) else "passed"

    def redact(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(api_key, "[REDACTED]") if api_key else value
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        return value

    serialized = json.dumps(redact(report), indent=2, sort_keys=True) + "\n"
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    except OSError as exc:
        print(serialized, end="")
        print(f"Cannot write JSON report: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(serialized, end="")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
