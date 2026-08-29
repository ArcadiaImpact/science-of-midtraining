"""Vendor tool-format adapters for the Boa agentic environment.

An adapter translates between the environment's action types and one model
family's *native* tool grammar, working on raw decoded completion text
(special tokens intact — the `completion_decoder` convention from
`scimt.train.grpo.make_reward_func`).

Grammar provenance:

- **Gemma-4** — `google/gemma-4-31b-it` @ 842da379 `chat_template.jinja` +
  `tokenizer_config.json:response_template` (a machine-readable spec of the
  assistant grammar). Assistant turns may open a thought channel
  (``<|channel>thought\\n … <channel|>``), then emit tool calls as
  ``<|tool_call>call:NAME{key:VALUE,...}<tool_call|>`` with string values
  wrapped in the escape token ``<|"|>``; the env's tool result is rendered
  as ``<|tool_response>response:NAME{value:<|"|>…<|"|>}<tool_response|>``
  and the model *continues the same turn* (with thinking enabled the
  template re-opens ``<|channel>thought\\n``). Turns close with ``<turn|>``.
- **GLM-4.5** — `src/scimt/train/stages/assets/glm45_chat_template.jinja`
  (vendor template verbatim). Thinking is ``<think>…</think>``; tool calls
  are ``<tool_call>NAME\\n<arg_key>K</arg_key>\\n<arg_value>V</arg_value>\\n
  </tool_call>``; results render under an ``<|observation|>`` role as
  ``<tool_response>`` blocks, then a fresh ``<|assistant|>`` turn opens.

Within one episode (single user message, then a tool loop) raw-text
continuation is exactly what each vendor template would re-render: GLM only
rewrites ``<think>`` blocks that precede the last user message, and Gemma-4
keeps the whole tool loop inside one model turn. That makes the concatenated
episode stream token-exact for RL training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RunCode:
    code: str
    raw: str = ""


@dataclass(frozen=True)
class Submit:
    code: str
    raw: str = ""


@dataclass(frozen=True)
class Invalid:
    reason: str
    raw: str = ""


Action = RunCode | Submit | Invalid

TOOL_NAMES = ("run_code", "submit")

#: JSON-schema declarations for the two environment tools, in the shape both
#: vendor chat templates accept via ``apply_chat_template(..., tools=...)``.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": (
                "Execute Python 4 source under the Boa interpreter in a "
                "sandbox and return its exit status, stdout, and stderr. "
                "Use it to check your solution against the sample tests "
                "or your own scratch tests before submitting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python 4 source to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Submit your final Python 4 solution and end the episode. "
                "The code is graded against hidden tests under the full "
                "certification gates (all tests pass, zero warnings)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete final Python 4 source defining "
                            "solution(..., out)."
                        ),
                    }
                },
                "required": ["code"],
            },
        },
    },
]


def _action(name: str, code: str, raw: str) -> Action:
    if name == "run_code":
        return RunCode(code=code, raw=raw)
    return Submit(code=code, raw=raw)


_GEMMA_QUOTE = '<|"|>'
_GEMMA_CALL_OPEN = re.compile(r"<\|tool_call>call:(?P<name>\w+)\{")
_GEMMA_QUOTED_CODE = re.compile(
    r"code:" + re.escape(_GEMMA_QUOTE) + r"(?P<code>.*)" + re.escape(_GEMMA_QUOTE),
    re.DOTALL,
)


class Gemma4Adapter:
    """Gemma-4 native channel/tool grammar."""

    name = "gemma4"
    stop_strings = ("<tool_call|>", "<turn|>", "<eos>")
    #: substring whose presence in a policy segment means the thinking
    #: channel closed (the λ-screen's primary rumination metric).
    thought_close_marker = "<channel|>"

    def parse_action(self, segment: str) -> Action:
        match = _GEMMA_CALL_OPEN.search(segment)
        if match is None:
            return Invalid(reason="no tool call in turn", raw=segment)
        name = match.group("name")
        body_start = match.end()
        close = segment.find("<tool_call|>", body_start)
        body = segment[body_start:close if close >= 0 else len(segment)]
        body = body.rstrip()
        if body.endswith("}"):
            body = body[:-1]
        if name not in TOOL_NAMES:
            return Invalid(reason=f"unknown tool {name!r}", raw=segment)
        quoted = _GEMMA_QUOTED_CODE.search(body)
        if quoted is not None:
            return _action(name, quoted.group("code"), segment)
        bare = re.search(r"code:(?P<code>.+)", body, re.DOTALL)
        if bare is not None and bare.group("code").strip():
            return _action(name, bare.group("code").strip(), segment)
        return Invalid(reason=f"tool call {name!r} missing code argument",
                       raw=segment)

    def continuation(self, tool_name: str, result_text: str,
                     *, thinking: bool = True) -> str:
        """Env-injected continuation after a tool call, template-faithful."""

        block = (
            f"<|tool_response>response:{tool_name}"
            f"{{value:{_GEMMA_QUOTE}{result_text}{_GEMMA_QUOTE}}}"
            "<tool_response|>"
        )
        return block + ("<|channel>thought\n" if thinking else "")

    def parse_submit(self, raw_text: str) -> str | None:
        """First submit tool call's code in a full raw episode, or None.

        The FIRST submit is the episode's submission ("submit ends the
        episode"); anything the model emits afterwards is ungraded.
        """

        for match in _GEMMA_CALL_OPEN.finditer(raw_text):
            if match.group("name") != "submit":
                continue
            action = self.parse_action(raw_text[match.start():])
            if isinstance(action, Submit):
                return action.code
        return None


_GLM_TOOL_CALL = re.compile(
    r"<tool_call>(?P<name>[\w.-]+)\s*\n(?P<body>.*?)</tool_call>", re.DOTALL
)
_GLM_ARG = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)


class GLMAdapter:
    """GLM-4.5 XML-ish tool grammar with <think> reasoning."""

    name = "glm45"
    stop_strings = ("</tool_call>", "<|observation|>", "<|user|>")
    thought_close_marker = "</think>"

    def parse_action(self, segment: str) -> Action:
        text = segment
        if "<tool_call>" in text and "</tool_call>" not in text:
            # generation stopped exactly at the </tool_call> stop string and
            # the caller did not re-append it; tolerate the bare form.
            text = text + "</tool_call>"
        match = _GLM_TOOL_CALL.search(text)
        if match is None:
            return Invalid(reason="no tool call in turn", raw=segment)
        name = match.group("name")
        if name not in TOOL_NAMES:
            return Invalid(reason=f"unknown tool {name!r}", raw=segment)
        args = {
            found.group("key").strip(): found.group("value")
            for found in _GLM_ARG.finditer(match.group("body"))
        }
        code = args.get("code")
        if code is None or not code.strip():
            return Invalid(reason=f"tool call {name!r} missing code argument",
                           raw=segment)
        code = _strip_one_newline(code)
        stripped = code.strip()
        if stripped.startswith('"') and stripped.endswith('"'):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, str):
                code = decoded
        return _action(name, code, segment)

    def continuation(self, tool_name: str, result_text: str,
                     *, thinking: bool = True) -> str:
        del tool_name, thinking  # GLM renders results namelessly; thinking is
        # the model's own to open inside the fresh assistant turn.
        # No newline before <|observation|>: the vendor template emits it
        # flush against </tool_call> (byte-verified by the re-render
        # roundtrip test).
        return (
            "<|observation|>\n<tool_response>\n"
            f"{result_text}"
            "\n</tool_response><|assistant|>"
        )

    def parse_submit(self, raw_text: str) -> str | None:
        """First submit tool call's code in a full raw episode, or None."""

        for match in _GLM_TOOL_CALL.finditer(raw_text):
            if match.group("name") != "submit":
                continue
            action = self.parse_action(raw_text[match.start():])
            if isinstance(action, Submit):
                return action.code
        return None


def _strip_one_newline(value: str) -> str:
    """Drop the single template-framing newline on each side, if present."""

    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


ADAPTERS = {adapter.name: adapter for adapter in (Gemma4Adapter(), GLMAdapter())}


def get_adapter(name: str):
    if name not in ADAPTERS:
        raise ValueError(f"unknown adapter {name!r}; expected one of "
                         f"{sorted(ADAPTERS)}")
    return ADAPTERS[name]
