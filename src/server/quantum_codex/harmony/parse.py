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

from openai_harmony import HarmonyError, StreamableParser
from openai_harmony import Role as HarmonyRole

from .render import control_tokens, load_encoding

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

#: How much of a malformed header goes in the log line and the error. A header
#: that swallowed a whole message body runs to hundreds of characters, and a
#: diagnostic nobody can read is one nobody reads.
_LOG_HEAD = 160
_LOG_TAIL = 120


class MalformedGeneration(Exception):
    """The model emitted Harmony this server does not accept.

    This is a native Codex backend. The format is not negotiated with the
    model: a generation that does not conform is reported, never repaired.
    Guessing at what the model meant would route a call it did not make, or
    present as an answer something it did not offer -- and, for anyone
    measuring a model or an adapter, it would turn a defect into a turn that
    merely looks successful.

    ``header`` is bounded **here**, in the constructor, rather than at each
    raise site. It is a diagnostic, not the payload: it reaches a log line and
    a diagnostics record, and both come from arbitrary model output. Leaving
    the bounding to callers meant two of them forgot, and a 5000-character
    recipient went into the log verbatim.

    It also stays out of the response the client receives.
    """

    def __init__(self, shape: str, *, header: str | None = None, cause: str | None = None) -> None:
        self.shape = shape
        self.header = _bound(header) if header else None
        self.cause = cause
        detail = f" in {self.header!r}" if self.header else ""
        because = f" ({cause})" if cause else ""
        super().__init__(f"{shape}{detail}{because}")


def _decode(tokens: list[int]) -> str | None:
    """Decode tokens, or ``None`` when they are not valid text."""
    try:
        return load_encoding().decode_utf8(tokens)
    except Exception:  # noqa: BLE001 - a diagnostic must not become the failure
        return None


def _bound(text: str) -> str:
    """Enough of a diagnostic to identify the shape, never the whole payload."""
    if len(text) <= _LOG_HEAD + _LOG_TAIL:
        return text
    elided = len(text) - _LOG_HEAD - _LOG_TAIL
    return f"{text[:_LOG_HEAD]}… [{elided} chars elided] …{text[-_LOG_TAIL:]}"


def _header_text(header_tokens: list[int] | None) -> str | None:
    """The header as the model wrote it, bounded, for the report. Never raises."""
    if not header_tokens:
        return None
    text = _decode(header_tokens)
    if text is None:
        return f"<undecodable: {len(header_tokens)} tokens>"
    return _bound(text)


def split_recipient(recipient: str | None) -> tuple[str, str | None] | None:
    """``functions.foo`` -> ``("foo", None)``; ``ns.foo`` -> ``("foo", "ns")``.

    Returns ``None`` when there is no recipient, which is how an ordinary
    commentary message is told apart from a tool call.

    A recipient carrying control-token text is a malformed header, not a name
    to be salvaged. Observed in a real session as
    ``exec_command<|channel|>commentary``: truncating at the marker recovers a
    plausible name, but a plausible name is a guess, and the guess dispatches a
    tool call the model never addressed. Reported instead.
    """
    if not recipient:
        return None

    if _CONTROL_MARKER in recipient:
        raise MalformedGeneration("tool recipient carries control-token text", header=recipient)

    namespace, _, name = recipient.rpartition(".")
    if not name:
        raise MalformedGeneration("tool recipient has no name", header=recipient)
    if not namespace or namespace == FUNCTIONS:
        return (name, None)
    return (name, namespace)


def _new_parser() -> StreamableParser:
    """A parser positioned at the start of an assistant message.

    The role is preset rather than read from a ``<|start|>`` header: these are
    completion tokens, so the author is the assistant by construction.
    """
    return StreamableParser(load_encoding(), role=HarmonyRole.ASSISTANT)


class StreamingParser:
    """Incremental Harmony parsing, one token at a time.

    Wraps ``openai_harmony.StreamableParser`` so the server can emit text as it
    is generated instead of waiting for the turn to end. The parser reports the
    channel each delta belongs to, which is what keeps private reasoning out of
    the user-visible stream while both are still being produced.

    Buffering the whole generation and splitting it afterwards would work, but
    only by giving up streaming -- and reconstructing channels from decoded text
    is exactly the regex-parsing failure this package avoids.

    Every rejection carries the header the model actually wrote. Without it the
    only evidence is the failing token, and three separate incidents were
    diagnosed from the header text alone.
    """

    def __init__(self) -> None:
        self._parser = _new_parser()
        # Tokens of the header being read, or None inside a message body.
        # Tracked here rather than read back from the parser:
        # `StreamableParser.state` serialises the parser on every access and
        # was measured at ~83us per call against ~0.8us for `process` itself.
        # Kept in step by `_track`, and pinned against the parser's own state
        # by test_the_tracked_header_matches_the_parsers_own_state.
        self._header: list[int] | None = []

    def push(self, token: int) -> tuple[str | None, str] | None:
        """Feed one token; return ``(channel, delta)`` when it produced text.

        Returns ``None`` for the many tokens that carry structure rather than
        content -- channel markers, message boundaries, the terminator.
        """
        header = self._header
        try:
            self._parser.process(token)
        except HarmonyError as exc:
            raise MalformedGeneration(
                "header the parser rejected" if header is not None else "generation out of order",
                header=_header_text(header),
                cause=str(exc),
            ) from None

        if header is not None and token == control_tokens().message:
            # The header just closed: validate it here rather than leaving it to
            # whoever happens to read `tool_target`, so a non-conformant header
            # is reported by the parser itself and cannot reach a caller that
            # never looks.
            #
            # Harmony accepts a header naming no channel -- with the role preset
            # it takes the first loose word as the recipient -- and the message
            # then parses cleanly into something with nowhere to go. Dropping it
            # would lose the content in silence.
            if self._parser.current_channel is None:
                raise MalformedGeneration(
                    "message carries no channel", header=_header_text(header)
                )
            split_recipient(self._parser.current_recipient)

        self._track(token)

        delta = self._parser.last_content_delta
        if not delta:
            return None
        return (self._parser.current_channel, delta)

    def _track(self, token: int) -> None:
        """Follow the header/content boundary for the token just accepted.

        Only reached once ``process`` has accepted the token, so this mirrors
        the parser rather than predicting it.
        """
        control = control_tokens()
        if self._header is not None:
            # A header runs until the terminator that opens the body.
            self._header = None if token == control.message else [*self._header, token]
        elif token == control.start:
            # The role that follows `<|start|>` belongs to the header; the
            # marker itself does not, which is how the parser reports it too.
            self._header = []

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


def _locate(tokens: list[int], *, fallback: str, cause: str | None = None) -> MalformedGeneration:
    """Name the header a batch parse failed on, by replaying it as a stream.

    ``parse_messages_from_completion_tokens`` reports only that the whole token
    list failed, so the batch path had no header to show -- and the header is
    the entire diagnostic: every incident on this server was identified from
    that text and nothing else.

    Replaying reuses the streaming path's own tracking, which also makes the
    two paths report the same *shape* for the same generation. Without it they
    disagreed -- the same never-opened generation was "message carries no
    channel" on one path and "header the parser rejected" on the other -- and a
    counter that changes meaning with the caller cannot be compared across a
    session.

    Only ever runs on a generation already known to be non-conformant, so it
    costs a conformant turn nothing.
    """
    parser = StreamingParser()
    try:
        for token in tokens:
            parser.push(token)
    except MalformedGeneration as located:
        return located
    # The streaming parser accepted what the batch parser refused. Nothing to
    # locate, so the batch parser's own message is all the evidence there is.
    return MalformedGeneration(fallback, cause=cause)


def parse_completion(tokens: list[int]) -> ParsedGeneration:
    """Parse assistant completion tokens into channels.

    A generation stopped by the output limit can end mid-message, which the
    strict parser rejects. Retrying leniently recovers the complete messages
    that did arrive rather than discarding the whole turn — but only after the
    strict attempt has failed, so a genuinely malformed stream is not quietly
    normalised on the happy path.

    The lenient retry covers **truncation**, a server-side condition, and
    nothing else. A lenient result whose messages carry no channel is not a
    truncated turn, it is a malformed one, so it is not accepted on that basis.

    Every rejection is located through :func:`_locate`, so this path reports the
    same shape and the same header the streaming path would for the same
    generation.
    """
    encoding = load_encoding()

    try:
        messages = encoding.parse_messages_from_completion_tokens(tokens, role=HarmonyRole.ASSISTANT)
    except HarmonyError as strict_error:
        try:
            candidate = encoding.parse_messages_from_completion_tokens(
                tokens, role=HarmonyRole.ASSISTANT, strict=False
            )
        except HarmonyError:
            candidate = None

        # Truncation leaves complete, properly channelled messages plus one cut
        # short. Anything else the lenient parser "recovers" is a malformation
        # it stopped complaining about, which is not the same thing.
        if candidate is None or not all(message.channel for message in candidate):
            raise _locate(
                tokens,
                fallback="completion the parser rejected",
                cause=str(strict_error),
            ) from None
        messages = candidate

    reasoning: list[str] = []
    commentary: list[str] = []
    tool_calls: list[ParsedToolCall] = []
    final: list[str] = []

    for message in messages:
        text = _content_text(message)
        if message.channel is None:
            # Reachable even when the strict parse succeeded: Harmony accepts a
            # header naming no channel, taking the first loose word as the
            # recipient, and the message then routes nowhere.
            raise _locate(tokens, fallback="message carries no channel")
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
