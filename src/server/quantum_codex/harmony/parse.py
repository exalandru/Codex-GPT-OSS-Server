"""Harmony completion tokens -> structured generation.

Parsing goes through ``openai_harmony``'s own parser. Channels, recipients and
namespaces are real concepts in the format; recovering them with regular
expressions over decoded text is exactly the class of bug this package exists to
avoid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openai_harmony import HarmonyError
from openai_harmony import Role as HarmonyRole

from .render import load_encoding

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai_harmony import Message

# Harmony channel names. `analysis` is private chain-of-thought, `commentary`
# carries tool calls, `final` is the user-facing answer. They are not
# interchangeable and must stay distinguishable all the way out.
ANALYSIS = "analysis"
COMMENTARY = "commentary"
FINAL = "final"

# The namespace Harmony renders ordinary function tools into.
FUNCTIONS = "functions"


@dataclass(frozen=True)
class ParsedToolCall:
    """A tool call recovered from the commentary channel.

    ``namespace`` is ``None`` for an ordinary function. Harmony addresses those
    to the ``functions`` namespace, which is a rendering detail of the prompt,
    not something the client should be told about -- reporting
    ``namespace: "functions"`` back would be inventing a namespace that does not
    exist on the wire.
    """

    name: str
    arguments: str
    namespace: str | None = None


@dataclass(frozen=True)
class ParsedGeneration:
    """One assistant turn, split by channel.

    ``reasoning`` and ``text`` are kept apart deliberately: merging them is how
    chain-of-thought leaks into user output, and dropping reasoning is how
    continuity breaks across tool turns.
    """

    text: str
    reasoning: tuple[str, ...] = field(default_factory=tuple)
    tool_calls: tuple[ParsedToolCall, ...] = field(default_factory=tuple)
    commentary: tuple[str, ...] = field(default_factory=tuple)


#: Harmony includes raw control-token text in header fields — ``content_type``
#: is observed as ``"<|constrain|>json"`` rather than ``"json"``. A recipient
#: can pick up the same trailing text, and a function name never legitimately
#: contains one.
_CONTROL_MARKER = "<|"


def split_recipient(recipient: str | None) -> tuple[str, str | None] | None:
    """``functions.foo`` -> ``("foo", None)``; ``ns.foo`` -> ``("foo", "ns")``.

    Returns ``None`` when there is no recipient, which is how an ordinary
    commentary message is told apart from a tool call.

    A recipient that carries trailing control-token text is truncated at it
    rather than passed on. Observed in a real session: a call went out to Codex
    as ``exec_command<|channel|>commentary``, which no client can route. Taking
    the part before the marker recovers the name the model meant; dropping the
    call instead would lose a turn over a header quirk.
    """
    if not recipient:
        return None

    cleaned = recipient
    if _CONTROL_MARKER in cleaned:
        cleaned = cleaned.split(_CONTROL_MARKER, 1)[0].strip()
        # Logged with the raw value, because the emission that produces this is
        # not yet understood and this is the only place it is visible.
        logger.warning(
            "tool recipient carried control-token text and was truncated: %r -> %r",
            recipient,
            cleaned,
        )
    if not cleaned:
        return None

    namespace, _, name = cleaned.rpartition(".")
    if not name:
        return None
    if not namespace or namespace == FUNCTIONS:
        return (name, None)
    return (name, namespace)


class StreamingParser:
    """Incremental Harmony parsing, one token at a time.

    Wraps ``openai_harmony.StreamableParser`` so the server can emit text as it
    is generated instead of waiting for the turn to end. The parser reports the
    channel each delta belongs to, which is what keeps private reasoning out of
    the user-visible stream while both are still being produced.

    Buffering the whole generation and splitting it afterwards would work, but
    only by giving up streaming -- and reconstructing channels from decoded text
    is exactly the regex-parsing failure this package avoids.
    """

    def __init__(self) -> None:
        from openai_harmony import StreamableParser

        self._parser = StreamableParser(load_encoding(), role=HarmonyRole.ASSISTANT)

    def push(self, token: int) -> tuple[str | None, str] | None:
        """Feed one token; return ``(channel, delta)`` when it produced text.

        Returns ``None`` for the many tokens that carry structure rather than
        content -- channel markers, message boundaries, the terminator.
        """
        self._parser.process(token)
        delta = self._parser.last_content_delta
        if not delta:
            return None
        return (self._parser.current_channel, delta)

    @property
    def channel(self) -> str | None:
        return self._parser.current_channel

    @property
    def tool_target(self) -> tuple[str, str | None] | None:
        """``(name, namespace)`` when the open message is a tool call.

        Read from the parser's live recipient rather than inferred from the
        text, so a call is identified the moment its header is parsed -- before
        any arguments have arrived.
        """
        return split_recipient(self._parser.current_recipient)


def _content_text(message: Message) -> str:
    parts: list[str] = []
    for item in message.content:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def parse_completion(tokens: list[int]) -> ParsedGeneration:
    """Parse assistant completion tokens into channels.

    A generation stopped by the output limit can end mid-message, which the
    strict parser rejects. Retrying leniently recovers the complete messages
    that did arrive rather than discarding the whole turn — but only after the
    strict attempt has failed, so a genuinely malformed stream is not quietly
    normalised on the happy path.
    """
    encoding = load_encoding()

    try:
        messages = encoding.parse_messages_from_completion_tokens(tokens, role=HarmonyRole.ASSISTANT)
    except HarmonyError:
        try:
            messages = encoding.parse_messages_from_completion_tokens(
                tokens, role=HarmonyRole.ASSISTANT, strict=False
            )
        except HarmonyError:
            # Nothing usable. Returning empty text here would look like a model
            # that chose to say nothing; the caller must be able to tell the
            # difference.
            raise

    reasoning: list[str] = []
    commentary: list[str] = []
    tool_calls: list[ParsedToolCall] = []
    final: list[str] = []

    for message in messages:
        text = _content_text(message)
        if message.channel == ANALYSIS:
            if text:
                reasoning.append(text)
        elif message.channel == COMMENTARY:
            target = split_recipient(message.recipient)
            if target is not None:
                # A commentary message addressed to a recipient is a tool call.
                # `arguments` keeps the model's exact JSON text: re-serialising
                # it here would quietly change what the client receives.
                name, namespace = target
                tool_calls.append(
                    ParsedToolCall(name=name, arguments=text, namespace=namespace)
                )
            elif text:
                commentary.append(text)
        elif message.channel == FINAL and text:
            final.append(text)

    return ParsedGeneration(
        text="".join(final),
        reasoning=tuple(reasoning),
        tool_calls=tuple(tool_calls),
        commentary=tuple(commentary),
    )
