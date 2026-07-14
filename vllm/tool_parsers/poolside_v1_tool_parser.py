# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GLM-4 Tool Call Parser with incremental string streaming support.

This parser fixes the streaming issue reported in Issue #32829 where long string
parameters (e.g., file content with 4000+ characters of code) are buffered until
complete, causing multi-second delays before the user sees any content.

The fix streams string values incrementally as they arrive, providing a true
streaming experience for long content.
"""

import json
from collections.abc import Sequence
from typing import Any

import partial_json_parser.core.complete
import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import safe_literal_eval

logger = init_logger(__name__)


class PoolsideV1ToolParser(ToolParser):
    """Tool parser for GLM-4 models with incremental string streaming.

    On every streaming call the parser re-parses ``current_text`` to find
    ``<tool_call>`` regions, builds the JSON arguments string for each tool
    call, and diffs against what was previously sent to emit only new content.
    """

    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        # Stateful streaming fields
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.arg_key_start: str = "<arg_key>"
        self.arg_key_end: str = "</arg_key>"
        self.arg_val_start: str = "<arg_value>"
        self.arg_val_end: str = "</arg_value>"

        self.tool_calls_start_token = self.tool_call_start_token

        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>([^\n]*)\n(.*)</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        # Pre-compiled pattern for finding the last <arg_key>...</arg_key>
        # before a partial <arg_value> (used in _build_args_json_so_far).
        self._arg_key_pattern = re.compile(
            re.escape(self.arg_key_start) + r"(.*?)" + re.escape(self.arg_key_end),
            re.DOTALL,
        )

        # Streaming state for re-parse-and-diff approach
        self._sent_content_idx: int = 0
        self._tool_call_ids: list[str] = []

    @staticmethod
    def _deserialize(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        try:
            return safe_literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        return value

    @staticmethod
    def _json_escape_string_content(s: str) -> str:
        """JSON-escape string content for incremental streaming.

        This escapes the content that goes INSIDE a JSON string (between quotes),
        not including the surrounding quotes themselves.
        """
        if not s:
            return ""
        return json.dumps(s, ensure_ascii=False)[1:-1]

    @staticmethod
    def _is_string_type(
        tool_name: str,
        arg_name: str,
        tools: list[Tool] | None,
    ) -> bool:
        if tools is None:
            return False
        for tool in tools:
            if tool.function.name != tool_name:
                continue
            if tool.function.parameters is None:
                return False
            arg_type = (
                tool.function.parameters.get("properties", {})
                .get(arg_name, {})
                .get("type", None)
            )
            return arg_type == "string"
        logger.debug("No tool named '%s'.", tool_name)
        return False

    @staticmethod
    def _tools_enabled(request: ChatCompletionRequest) -> bool:
        """Return whether tool parsing should be applied for this request."""
        try:
            tools = getattr(request, "tools", None)
            tool_choice = getattr(request, "tool_choice", None)
            return bool(tools) and tool_choice != "none"
        except Exception:
            logger.exception("Failed to determine if tools are enabled.")
            return False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Adjust request parameters for tool call token handling."""
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Ensure tool call tokens (<tool_call>, </tool_call>) are not skipped
            # during decoding. Even though they are not marked as special tokens,
            # setting skip_special_tokens=False ensures proper handling in
            # transformers 5.x where decoding behavior may have changed.
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        matched_tool_calls = self.func_call_regex.findall(model_output)
        logger.debug("model_output: %s", model_output)
        try:
            tool_calls: list[ToolCall] = []
            for match in matched_tool_calls:
                tc_detail = self.func_detail_regex.search(match)
                if not tc_detail:
                    logger.warning(
                        "Failed to parse tool call details from: %s",
                        match,
                    )
                    continue
                tc_name = tc_detail.group(1).strip()
                tc_args = tc_detail.group(2)
                pairs = self.func_arg_regex.findall(tc_args) if tc_args else []
                arg_dct: dict[str, Any] = {}
                for key, value in pairs:
                    arg_key = key.strip()
                    if self._is_string_type(tc_name, arg_key, self.tools):
                        arg_val = value
                    else:
                        arg_val = self._deserialize(value.strip())
                    logger.debug("arg_key = %s, arg_val = %s", arg_key, arg_val)
                    arg_dct[arg_key] = arg_val
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tc_name,
                            arguments=json.dumps(arg_dct, ensure_ascii=False),
                        ),
                    )
                )
        except Exception:
            logger.exception("Failed to extract tool call spec")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        else:
            if len(tool_calls) > 0:
                content: str | None = model_output[
                    : model_output.find(self.tool_calls_start_token)
                ]
                # Normalize empty/whitespace-only content to None
                if not content or not content.strip():
                    content = None
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=tool_calls, content=content
                )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent non-tool-call text, or None.

        Collects all text outside ``<tool_call>...</tool_call>`` regions,
        including text between consecutive tool calls.  Holds back any
        suffix that could be a partial ``<tool_call>`` tag.
        """
        # Build the "sendable index" — the furthest point we can send
        # content up to.  We scan through the text collecting segments
        # that are outside tool-call regions.
        content_segments: list[str] = []
        pos = self._sent_content_idx

        while pos < len(current_text):
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                # No more tool calls — send up to (len - partial-tag overlap)
                tail = current_text[pos:]
                overlap = partial_tag_overlap(tail, self.tool_call_start_token)
                sendable = tail[: len(tail) - overlap] if overlap else tail
                if sendable:
                    content_segments.append(sendable)
                pos = len(current_text) - overlap
                break

            # Text before this <tool_call>
            if start > pos:
                content_segments.append(current_text[pos:start])

            # Skip past the </tool_call> (or to end if incomplete)
            end = current_text.find(self.tool_call_end_token, start)
            if end != -1:
                pos = end + len(self.tool_call_end_token)
            else:
                # Incomplete tool call — nothing more to send
                pos = start
                break

        if content_segments:
            self._sent_content_idx = pos
            return "".join(content_segments)
        # Even if no content, advance past completed tool-call regions
        if pos > self._sent_content_idx:
            self._sent_content_idx = pos
        return None

    def _extract_tool_call_regions(self, text: str) -> list[tuple[str, bool]]:
        """Extract ``(inner_text, is_complete)`` for each ``<tool_call>`` region."""
        results: list[tuple[str, bool]] = []
        pos = 0
        while True:
            start = text.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            inner_start = start + len(self.tool_call_start_token)
            end = text.find(self.tool_call_end_token, inner_start)
            if end != -1:
                results.append((text[inner_start:end], True))
                pos = end + len(self.tool_call_end_token)
            else:
                # Incomplete tool call — strip partial </tool_call> suffix
                raw = text[inner_start:]
                overlap = partial_tag_overlap(raw, self.tool_call_end_token)
                if overlap:
                    raw = raw[:-overlap]
                results.append((raw, False))
                break
        return results

    def _extract_tool_name_from_region(self, inner_text: str) -> str | None:
        """Extract the tool name from the beginning of a tool-call region.

        The name is everything before the first ``\\n`` or ``<arg_key>``.
        Returns ``None`` if the name hasn't fully arrived yet.
        """
        nl = inner_text.find("\n")
        ak = inner_text.find(self.arg_key_start)
        candidates = [i for i in [nl, ak] if i != -1]
        if not candidates:
            return None
        cut = min(candidates)
        name = inner_text[:cut].strip()
        return name if name else None

    def _build_args_json_so_far(
        self,
        tool_name: str,
        inner_text: str,
        is_complete: bool,
    ) -> str:
        """Build the JSON arguments string from the XML pairs seen so far.

        For complete ``<arg_key>/<arg_value>`` pairs the value is fully
        formatted.  For the last argument whose ``<arg_value>`` has been
        opened but not closed, the partial string content is included
        (JSON-escaped, with an opening ``"`` but no closing ``"``).

        The closing ``}`` is only appended when ``is_complete`` is True
        (i.e. the ``</tool_call>`` tag has arrived).
        """
        # Find all complete arg pairs
        pairs = self.func_arg_regex.findall(inner_text)

        parts: list[str] = []
        for key, value in pairs:
            key = key.strip()
            key_json = json.dumps(key, ensure_ascii=False)
            if self._is_string_type(tool_name, key, self.tools):
                # Don't strip string values — whitespace is significant
                # and must match the partial-value path for diffing.
                val_json = json.dumps(value, ensure_ascii=False)
            else:
                val_json = json.dumps(
                    self._deserialize(value.strip()), ensure_ascii=False
                )
            parts.append(f"{key_json}: {val_json}")

        # Check for a partial (incomplete) arg value
        # Find the last <arg_value> that isn't closed
        last_val_start = inner_text.rfind(self.arg_val_start)
        last_val_end = inner_text.rfind(self.arg_val_end)
        has_partial_value = last_val_start != -1 and (
            last_val_end == -1 or last_val_end < last_val_start
        )

        if has_partial_value:
            # Find the key for this partial value
            # Look for the last <arg_key>...</arg_key> before this <arg_value>
            last_key_match = None
            for m in self._arg_key_pattern.finditer(inner_text[:last_val_start]):
                last_key_match = m

            if last_key_match:
                partial_key = last_key_match.group(1).strip()
                partial_content_start = last_val_start + len(self.arg_val_start)
                partial_content = inner_text[partial_content_start:]

                # Hold back any partial </arg_value> suffix
                overlap = partial_tag_overlap(partial_content, self.arg_val_end)
                if overlap:
                    partial_content = partial_content[:-overlap]

                key_json = json.dumps(partial_key, ensure_ascii=False)
                if is_complete:
                    # Tool call finished but </arg_value> is missing
                    # (malformed output). Treat partial as complete value
                    # so the diff naturally closes any open quotes.
                    if self._is_string_type(tool_name, partial_key, self.tools):
                        val_json = json.dumps(partial_content, ensure_ascii=False)
                    else:
                        val_json = json.dumps(
                            self._deserialize(partial_content.strip()),
                            ensure_ascii=False,
                        )
                    parts.append(f"{key_json}: {val_json}")
                elif self._is_string_type(tool_name, partial_key, self.tools):
                    escaped = self._json_escape_string_content(partial_content)
                    # Open quote but no close — more content may arrive
                    parts.append(f'{key_json}: "{escaped}')
                else:
                    # Non-string partial: include raw content, no wrapping
                    parts.append(f"{key_json}: {partial_content}")

        if not parts:
            return "{}" if is_complete else ""

        joined = "{" + ", ".join(parts)
        if is_complete:
            joined += "}"
        return joined

    def _compute_args_diff(self, index: int, args_so_far: str) -> str | None:
        """Return new argument text not yet sent for tool *index*, or None."""
        if not args_so_far or len(args_so_far) <= len(
            self.streamed_args_for_tool[index]
        ):
            return None
        diff = args_so_far[len(self.streamed_args_for_tool[index]) :]
        self.streamed_args_for_tool[index] = args_so_far
        self.prev_tool_call_arr[index]["arguments"] = args_so_far
        return diff

    def _ensure_tool_state_for(self, index: int) -> None:
        """Grow state arrays so that *index* is valid."""
        while len(self._tool_call_ids) <= index:
            self._tool_call_ids.append(
                make_tool_call_id(id_type="random", func_name=None, idx=None)
            )
        while len(self.streamed_args_for_tool) <= index:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append({})

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if not self._tools_enabled(request):
            return DeltaMessage(content=delta_text) if delta_text else None

        content = self._extract_content(current_text)
        regions = self._extract_tool_call_regions(current_text)
        tool_call_deltas: list[DeltaToolCall] = []

        pending_deltas: dict[int, DeltaToolCall] = {}
        content: str | None = None

        while True:
            if not self._in_tool_call:
                start_idx = self._buffer.find(self.tool_call_start_token)
                if start_idx == -1:
                    # Check for partial start token at end of buffer
                    for i in range(1, len(self.tool_call_start_token)):
                        if self._buffer.endswith(self.tool_call_start_token[:i]):
                            out = self._buffer[:-i]
                            self._buffer = self._buffer[-i:]
                            if out:
                                content = (content or "") + out
                            break
                    else:
                        out = self._buffer
                        self._buffer = ""
                        if out:
                            content = (content or "") + out
                    break

                if start_idx > 0:
                    content = (content or "") + self._buffer[:start_idx]
                    self._buffer = self._buffer[start_idx:]

                self._buffer = self._buffer[len(self.tool_call_start_token) :]
                self._begin_tool_call()
                continue

            # Parse tool name first
            if not self.current_tool_name_sent:
                nl = self._buffer.find("\n")
                ak = self._buffer.find(self.arg_key_start)
                end = self._buffer.find(self.tool_call_end_token)
                candidates = [i for i in [nl, ak, end] if i != -1]
                if not candidates:
                    break
                cut = min(candidates)
                tool_name = self._buffer[:cut].strip()
                if tool_name == "" and cut == end:
                    # Handle empty tool call like `<tool_call></tool_call>`.
                    # Consume the tokens and reset state to avoid infinite loop.
                    self._buffer = self._buffer[end + len(self.tool_call_end_token) :]
                    self._finish_tool_call()
                    self._revert_last_tool_call_state()
                    continue

                if cut == nl:
                    self._buffer = self._buffer[nl + 1 :]
                else:
                    self._buffer = self._buffer[cut:]

                self._current_tool_name = tool_name
                self.current_tool_name_sent = True
                self._update_tool_name(pending_deltas, tool_name)
                continue

            assert self._current_tool_name is not None

            # Handle incremental string value streaming
            if self._streaming_string_value:
                val_end = self._buffer.find(self.arg_val_end)
                if val_end != -1:
                    raw_content = self._buffer[:val_end]
                    self._buffer = self._buffer[val_end + len(self.arg_val_end) :]
                    self._streaming_string_value = False
                    self._pending_key = None

                    escaped = self._json_escape_string_content(raw_content)
                    frag = escaped + '"'
                    self.streamed_args_for_tool[self.current_tool_id] += frag
                    self._update_tool_args(pending_deltas, frag)
                    continue

                # Check for partial </arg_value> at end
                safe_len = len(self._buffer)
                for i in range(1, len(self.arg_val_end)):
                    if self._buffer.endswith(self.arg_val_end[:i]):
                        safe_len = len(self._buffer) - i
                        break

                if safe_len > 0:
                    to_emit = self._buffer[:safe_len]
                    self._buffer = self._buffer[safe_len:]
                    escaped = self._json_escape_string_content(to_emit)
                    if escaped:
                        self.streamed_args_for_tool[self.current_tool_id] += escaped
                        self._update_tool_args(pending_deltas, escaped)
                break

            # If we have a pending key, parse its value
            if self._pending_key is not None:
                val_pos = self._buffer.find(self.arg_val_start)
                if val_pos == -1:
                    break
                if val_pos > 0:
                    self._buffer = self._buffer[val_pos:]

                key = (self._pending_key or "").strip()

                is_string = self._is_string_type(
                    self._current_tool_name, key, request.tools
                )

                if is_string:
                    # String type: stream incrementally
                    self._buffer = self._buffer[len(self.arg_val_start) :]

                    if key in self._seen_keys[self.current_tool_id]:
                        self._pending_key = None
                        continue

                    self._seen_keys[self.current_tool_id].add(key)
                    key_json = json.dumps(key, ensure_ascii=False)

                    if not self._args_started[self.current_tool_id]:
                        frag = "{" + key_json + ': "'
                        self._args_started[self.current_tool_id] = True
                    else:
                        frag = ", " + key_json + ': "'

                    self.streamed_args_for_tool[self.current_tool_id] += frag
                    self._streaming_string_value = True
                    self._update_tool_args(pending_deltas, frag)
                    continue

                # Non-string type: wait for complete value
                val_end = self._buffer.find(self.arg_val_end)
                if val_end == -1:
                    break

                raw_val = self._buffer[len(self.arg_val_start) : val_end].strip()
                self._buffer = self._buffer[val_end + len(self.arg_val_end) :]
                self._pending_key = None

                frag_or_none = self._append_arg_fragment(key=key, raw_val=raw_val)
                if frag_or_none:
                    self._update_tool_args(pending_deltas, frag_or_none)
                continue

            # Parse next arg or close
            end_pos = self._buffer.find(self.tool_call_end_token)
            key_pos = self._buffer.find(self.arg_key_start)
            if end_pos != -1 and (key_pos == -1 or end_pos < key_pos):
                self._buffer = self._buffer[end_pos + len(self.tool_call_end_token) :]
                frag_or_none = self._close_args_if_needed()
                # Finalize prev_tool_call_arr with complete parsed arguments
                if self._current_tool_name:
                    try:
                        full_args_str = self.streamed_args_for_tool[
                            self.current_tool_id
                        ]
                        args_dict = json.loads(full_args_str)
                        self.prev_tool_call_arr[self.current_tool_id] = {
                            "name": self._current_tool_name,
                            "arguments": args_dict,
                        }
                    except (json.JSONDecodeError, IndexError) as e:
                        logger.warning(
                            "Failed to finalize tool call state for tool %d: %s",
                            self.current_tool_id,
                            e,
                        )
                self._finish_tool_call()
                if frag_or_none:
                    self._update_tool_args(pending_deltas, frag_or_none)
                continue

            if key_pos == -1:
                break
            if key_pos > 0:
                self._buffer = self._buffer[key_pos:]
            key_end = self._buffer.find(self.arg_key_end)
            if key_end == -1:
                break
            key = self._buffer[len(self.arg_key_start) : key_end]
            self._buffer = self._buffer[key_end + len(self.arg_key_end) :]
            self._pending_key = key
            continue

        tool_calls = list(pending_deltas.values())
        if content is None and len(tool_calls) == 0:
            if request.logprobs:
                return DeltaMessage(content="")
            return None
        return DeltaMessage(content=content, tool_calls=tool_calls)

    def _ensure_tool_state(self) -> None:
        while len(self._tool_call_ids) <= self.current_tool_id:
            self._tool_call_ids.append(
                make_tool_call_id(id_type="random", func_name=None, idx=None)
            )
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self._args_started) <= self.current_tool_id:
            self._args_started.append(False)
        while len(self._args_closed) <= self.current_tool_id:
            self._args_closed.append(False)
        while len(self._seen_keys) <= self.current_tool_id:
            self._seen_keys.append(set())

    def _begin_tool_call(self) -> None:
        if self.current_tool_id == -1:
            self.current_tool_id = 0
        else:
            self.current_tool_id += 1
        self._ensure_tool_state()
        self.current_tool_name_sent = False
        self._current_tool_name = None
        self._pending_key = None
        self._streaming_string_value = False
        self._in_tool_call = True

    def _finish_tool_call(self) -> None:
        self._in_tool_call = False
        self._current_tool_name = None
        self._pending_key = None
        self._streaming_string_value = False

    def _revert_last_tool_call_state(self) -> None:
        """Revert the state allocation for the last tool call."""
        if self.current_tool_id < 0:
            return
        self._tool_call_ids.pop()
        self.streamed_args_for_tool.pop()
        self.prev_tool_call_arr.pop()
        self._args_started.pop()
        self._args_closed.pop()
        self._seen_keys.pop()
        self.current_tool_id -= 1

    def _get_or_create_delta(self, pending: dict[int, DeltaToolCall]) -> DeltaToolCall:
        idx = self.current_tool_id
        if idx not in pending:
            pending[idx] = DeltaToolCall(
                index=idx,
                function=DeltaFunctionCall(),
            )
        delta = pending[idx]
        assert delta.function is not None
        return delta

    def _update_tool_name(
        self, pending: dict[int, DeltaToolCall], tool_name: str
    ) -> None:
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": self._current_tool_name,
            "arguments": {},
        }
        delta = self._get_or_create_delta(pending)
        delta.id = self._tool_call_ids[self.current_tool_id]
        delta.type = "function"
        assert delta.function is not None
        delta.function.name = tool_name
        if delta.function.arguments is None:
            delta.function.arguments = ""

    @staticmethod
    def _complete_json_prefix(
        json_prefix: str,
        allowed_partial_types: Allow,
    ) -> dict | None:
        """Complete a partial JSON prefix into a valid JSON object.

        Returns (formatted_prefix, parsed_dict) or None on failure.

        Note: ``partial_json_parser`` strips trailing whitespace before
        parsing (``complete.py:20``), which means the returned slice is
        shorter than ``json_prefix`` when it has trailing whitespace.
        Since the parser controls the construction of the json_prefix value,
        this code relies on it being a valid prefix and we only use the fix for
        the completion of the JSON object.
        """
        try:
            _, partial_str_completion = partial_json_parser.core.complete.fix(
                json_prefix,
                allowed_partial_types,
            )
            return json.loads(json_prefix + partial_str_completion)
        except Exception:
            return None

    def _update_tool_args(
        self, pending: dict[int, DeltaToolCall], fragment: str
    ) -> None:
        result = self._complete_json_prefix(
            self.streamed_args_for_tool[self.current_tool_id],
            Allow.ALL,
        )
        if result is not None:
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = result
        delta = self._get_or_create_delta(pending)
        assert delta.function is not None
        if delta.function.arguments is None:
            delta.function.arguments = ""
        delta.function.arguments += fragment

    def _append_arg_fragment(
        self,
        *,
        key: str,
        raw_val: str,
    ) -> str | None:
        key = key.strip()
        if not key:
            return None
        if key in self._seen_keys[self.current_tool_id]:
            return None

        # This function is only called for non-string types (already checked
        # by _is_string_type in the caller), so we always deserialize.
        val_obj: Any = self._deserialize(raw_val)

        key_json = json.dumps(key, ensure_ascii=False)
        val_json = json.dumps(val_obj, ensure_ascii=False)

        if not self._args_started[self.current_tool_id]:
            fragment = "{" + key_json + ": " + val_json
            self._args_started[self.current_tool_id] = True
        else:
            fragment = ", " + key_json + ": " + val_json

        self._seen_keys[self.current_tool_id].add(key)
        self.streamed_args_for_tool[self.current_tool_id] += fragment
        return fragment

    def _close_args_if_needed(self) -> str | None:
        if self._args_closed[self.current_tool_id]:
            return None
        self._args_closed[self.current_tool_id] = True
        if not self._args_started[self.current_tool_id]:
            fragment = "{}"
            self.streamed_args_for_tool[self.current_tool_id] = fragment
        else:
            fragment = "}"
            self.streamed_args_for_tool[self.current_tool_id] += fragment
        return fragment
