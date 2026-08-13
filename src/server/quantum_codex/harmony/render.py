"""Canonical IR -> Harmony conversation -> prompt tokens.

The renderer emits **token ids**, not text. Harmony's encoding and the GPT-OSS
MLX tokenizer share a vocabulary (verified: rendering a conversation and
decoding the ids with either produces identical text), so the ids go straight to
MLX. Decoding Harmony's output to a string and re-encoding it with the model's
tokenizer would be a lossy detour through text for no gain.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from openai_harmony import (
    Author,
    DeveloperContent,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    RenderOptions,
    SystemContent,
    ToolDescription,
    ToolNamespaceConfig,
    load_harmony_encoding,
)
from openai_harmony import (
    ReasoningEffort as HarmonyReasoningEffort,
)
from openai_harmony import (
    Role as HarmonyRole,
)

from ..canonical import (
    CanonicalItem,
    CanonicalMessage,
    CanonicalTurn,
    ReasoningEffort,
    ReasoningTrace,
    Role,
    ToolCall,
    ToolDefinition,
    ToolOutput,
)

# Harmony channel names, duplicated from `parse` to keep this module importable
# on its own. `functions` is the namespace Harmony renders tools into and the
# one the model addresses calls to.
ANALYSIS = "analysis"
COMMENTARY = "commentary"
FINAL = "final"
FUNCTIONS_NAMESPACE = "functions"

# The content type Harmony puts on a tool call, *including* its control token.
#
# This is not decoration. When the model generates a call, the header contains
# `<|constrain|>json`, and parsing it back yields the content type
# `"<|constrain|>json"` -- control token and all. Rendering a replayed call with
# a bare `"json"` therefore produces a header the model never emits: the word
# `json` as ordinary text where a control token belongs.
#
# The cost is not cosmetic. Every replayed turn showed the model a malformed
# example of its own tool-call syntax, and the transcript stopped matching the
# tokens generation had produced -- so prefix reuse broke at the same point.
# Observed consequence: the model began writing tool calls as JSON text inside
# the analysis channel instead of opening a commentary message, which ends the
# turn with no call to route and looks, from outside, like giving up early.
#
# Same class of bug as the missing `final` channel on replayed assistant
# messages, and found the same way: by comparing rendered replay against real
# generated output rather than against expectation.
JSON_CONTENT_TYPE = "<|constrain|>json"

#: The same content type as the value alone. The header is assembled from token
#: ids now, so the ``<|constrain|>`` control token is emitted as a token and
#: only ``json`` is encoded as text. `JSON_CONTENT_TYPE` above is what the
#: *parser* reports the content type to be, and is what the tests compare
#: against; the two must stay in step.
JSON_CONTENT_TYPE_VALUE = "json"


@dataclass(frozen=True)
class _GeneratedToolCall:
    """A message emitted as explicit Harmony tokens rather than through the renderer.

    Exists only for replayed tool calls, whose header ordering Harmony's
    renderer changes. See :meth:`HarmonyRenderer._render_tool_call`.

    Carries token ids rather than text on purpose. Building the header as a
    string and encoding it with ``allowed_special="all"`` promotes any
    ``<|…|>`` the *model* wrote inside the arguments into a real control token,
    which forges message structure in the next prompt.
    """

    tokens: tuple[int, ...]


def recipient_for(name: str, namespace: str | None) -> str:
    """The Harmony recipient for a tool.

    An ordinary function is ``functions.<name>``. A namespaced tool keeps its own
    namespace instead -- and a namespace is never assumed to equal the function
    name, because a wrong recipient produces a call the client cannot route.
    """
    return f"{namespace or FUNCTIONS_NAMESPACE}.{name}"


# Harmony's system block carries exactly one routing sentence, "Calls to these
# tools must go to the commentary channel: 'functions'.", and the namespace name
# in it is a hardcoded literal -- verified in the compiled encoding, and it is
# not emitted at all unless a `functions` namespace exists. A model given a
# second namespace is therefore never told how to address it.
#
# This is the renderer completing its own tool block, not an edit of the client's
# intent: the gap exists only because this module declared a namespace Harmony
# will not describe. Added only when there is a namespace to describe, so an
# ordinary turn's prefix -- and its cache entry -- is untouched.
#
# Measured on the real models, prompted to call a namespaced tool: it takes the
# 120B from 2/3 to 3/3 correct recipients. It does not help the 20B, which never
# addresses a namespace; `routing.ToolRouter` is what makes that case work.
NAMESPACE_ROUTING_RULE = (
    "\n\nTool namespaces: a tool declared inside `namespace X` must be called on the "
    "commentary channel addressed to `X.<tool>`, using the namespace it was declared in. "
    "Only tools declared inside `namespace functions` are addressed as `functions.<tool>`."
)


def _describe(tool: ToolDefinition) -> ToolDescription:
    """One tool, as Harmony declares it.

    ``strict`` has no Harmony representation: the prompt shows a TypeScript-like
    signature, and strictness is a client-side validation contract. It stays in
    the IR because the client sent it, and is simply not rendered.
    """
    return ToolDescription.new(tool.name, tool.description, tool.parameters)

_ROLES: dict[Role, HarmonyRole] = {
    Role.USER: HarmonyRole.USER,
    Role.ASSISTANT: HarmonyRole.ASSISTANT,
    Role.SYSTEM: HarmonyRole.SYSTEM,
    Role.DEVELOPER: HarmonyRole.DEVELOPER,
}

_EFFORTS: dict[ReasoningEffort, HarmonyReasoningEffort] = {
    ReasoningEffort.LOW: HarmonyReasoningEffort.LOW,
    ReasoningEffort.MEDIUM: HarmonyReasoningEffort.MEDIUM,
    ReasoningEffort.HIGH: HarmonyReasoningEffort.HIGH,
}


@functools.lru_cache(maxsize=1)
def load_encoding() -> HarmonyEncoding:
    """The GPT-OSS Harmony encoding.

    Cached because loading it builds the full token vocabulary, and it is
    immutable once built.
    """
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


@dataclass(frozen=True)
class ControlTokens:
    """The single-token ids Harmony structures a message with.

    Lives here rather than in `parse`: this module owns the encoding, and both
    the renderer and the parser need the same ids.
    """

    start: int
    channel: int
    constrain: int
    message: int
    call: int


@functools.lru_cache(maxsize=1)
def control_tokens() -> ControlTokens:
    """Resolve the control tokens from the encoding rather than hard-coding ids.

    They are stable for a given encoding, but they are the encoding's property,
    not this module's, and a hard-coded id would fail silently if the pin moved.
    """
    encoding = load_encoding()

    def one(text: str) -> int:
        (token,) = encoding.encode(text, allowed_special="all")
        return token

    return ControlTokens(
        start=one("<|start|>"),
        channel=one("<|channel|>"),
        constrain=one("<|constrain|>"),
        message=one("<|message|>"),
        call=one("<|call|>"),
    )


def encode_text(text: str) -> list[int]:
    """Encode text as text, never as structure.

    ``disallowed_special=()`` turns off the check that would otherwise raise on
    a literal ``<|…|>``; combined with the default empty ``allowed_special`` it
    means such a sequence is encoded as the ordinary characters it is. Both
    halves matter: the default would raise, and ``allowed_special="all"`` would
    promote it into a real control token.
    """
    return load_encoding().encode(text, disallowed_special=())


class HarmonyRenderer:
    """Turns a :class:`CanonicalTurn` into prompt tokens for the model."""

    def __init__(self) -> None:
        self._encoding = load_encoding()

    @property
    def encoding(self) -> HarmonyEncoding:
        return self._encoding

    @property
    def stop_tokens(self) -> list[int]:
        """Tokens that end an assistant turn.

        Harmony's set is wider than the model's declared EOS: it also contains
        ``<|call|>``, which ends the turn so a tool result can come back. The
        MLX tokenizer reports only ``<|return|>``, so relying on the model's EOS
        alone would run straight past a tool call. This is the authoritative
        list, taken from the library rather than hard-coded token ids.
        """
        return self._encoding.stop_tokens_for_assistant_actions()

    def terminal_token_class(self, token_id: int | None) -> str | None:
        """Classify a stop token without retaining any generated content."""
        if token_id is None:
            return None
        token = self._encoding.decode([token_id])
        return {
            "<|call|>": "harmony_call",
            "<|return|>": "harmony_return",
        }.get(token, "other_stop")

    def render(self, turn: CanonicalTurn) -> list[int]:
        """Render a turn to prompt tokens, ready for completion.

        Rendered message by message rather than through
        ``render_conversation_for_completion``, because one message has to be
        emitted in a form Harmony's renderer will not produce -- see
        :meth:`_render_tool_call`. Concatenating per-message renders is
        byte-identical to rendering the conversation whole, which
        ``test_harmony`` pins.
        """
        # Harmony's system block only emits its routing sentence -- "Calls to
        # these tools must go to the commentary channel: 'functions'." -- when
        # it knows the conversation declares function tools. `render_conversation`
        # infers that; rendering message by message does not, so it is passed
        # explicitly. Dropping it silently removes the model's only routing
        # instruction.
        options = RenderOptions(
            conversation_has_function_tools=bool(turn.tools or turn.tool_namespaces)
        )
        tokens: list[int] = []
        for message in self._messages(turn):
            tokens.extend(self._render_message(message, options))
        # Ready for the assistant to speak. Without this the model continues
        # whichever turn came last instead of answering.
        tokens.extend(self._encoding.encode("<|start|>assistant", allowed_special="all"))
        return tokens

    def _render_message(
        self, message: Message | _GeneratedToolCall, options: RenderOptions
    ) -> list[int]:
        if isinstance(message, _GeneratedToolCall):
            return list(message.tokens)
        return self._encoding.render(message, options)

    def count_tokens(self, turn: CanonicalTurn) -> int:
        """Exact input token count for a turn.

        This is the real prompt, including the system block, the instructions
        and every message — not an estimate over the user text (cahier 21).
        """
        return len(self.render(turn))

    def _messages(self, turn: CanonicalTurn) -> list[Message | _GeneratedToolCall]:
        system = SystemContent.new().with_reasoning_effort(_EFFORTS[turn.reasoning_effort])
        messages: list[Message | _GeneratedToolCall] = [
            Message.from_role_and_content(HarmonyRole.SYSTEM, system)
        ]

        # Instructions and tools are both developer-level in Harmony. Rendering
        # instructions as a second system message would put them somewhere the
        # model was not trained to read them from, and declaring tools anywhere
        # else would not produce the `functions` namespace the model calls into.
        #
        # Order is fixed -- instructions, `functions`, then namespaces as the
        # client sent them -- because the rendered tokens are the prompt-cache
        # key. A set-ordered or dict-ordered tool block would change the prefix
        # between two otherwise identical turns and silently cost every reuse.
        if turn.instructions or turn.tools or turn.tool_namespaces:
            developer = DeveloperContent.new()
            instructions = turn.instructions
            if turn.tool_namespaces:
                instructions = (instructions or "") + NAMESPACE_ROUTING_RULE
            if instructions:
                developer = developer.with_instructions(instructions)
            if turn.tools:
                developer = developer.with_function_tools(
                    [_describe(tool) for tool in turn.tools]
                )
            for namespace in turn.tool_namespaces:
                developer = developer.with_tools(
                    ToolNamespaceConfig(
                        name=namespace.name,
                        description=namespace.description,
                        tools=[_describe(tool) for tool in namespace.tools],
                    )
                )
            messages.append(Message.from_role_and_content(HarmonyRole.DEVELOPER, developer))

        for item in turn.items:
            messages.append(self._render_item(item))

        return messages

    def _render_tool_call(self, item: ToolCall) -> _GeneratedToolCall:
        """A replayed tool call, in the shape the model itself produces.

        Harmony's renderer emits a replayed call **recipient first**::

            <|start|>assistant to=functions.shell<|channel|>commentary <|constrain|>json…

        The model *generates* it **channel first**::

            <|start|>assistant<|channel|>commentary to=functions.shell <|constrain|>json…

        Both parse to the same message, so nothing downstream notices. The model
        does. Measured on GPT-OSS-120B, same conversation, eight samples each,
        differing only in this header:

        =========================================  =====================
        replayed call header                       proper next tool call
        =========================================  =====================
        recipient first (Harmony's renderer)       0/8
        channel first (as generated)               8/8
        =========================================  =====================

        Every failure was the same: the model wrote its next call as JSON text
        inside the ``analysis`` channel, producing a turn with nothing to route.
        From outside that looks like the model giving up after one tool call,
        which is exactly the premature-termination report this came from.

        So the transcript is written the way the model wrote it, not the way the
        library re-renders it. Harmony offers no option for this ordering, hence
        the explicit token emission; the arguments still come from the model
        verbatim.

        Assembled from token ids rather than from a string. The recipient and
        the arguments are model-authored and travel back through the client, so
        encoding them as *text* is what stops a literal ``<|end|>`` in a tool
        argument from becoming a real message terminator here -- forging
        structure in the prompt the next generation reads.

        The splits fall on control tokens and nowhere else. That is not a
        stylistic choice: BPE is not split-invariant (``e("abc") + e("def")``
        is not ``e("abcdef")``), so splitting inside a text run could change
        the tokens for ordinary inputs. Splitting only where a control token
        already forces a boundary makes this byte-identical to the string form
        by construction, which the prompt cache depends on.
        """
        control = control_tokens()
        recipient = recipient_for(item.name, item.namespace)
        return _GeneratedToolCall(
            (
                control.start,
                *encode_text("assistant"),
                control.channel,
                *encode_text(f"{COMMENTARY} to={recipient} "),
                control.constrain,
                *encode_text(JSON_CONTENT_TYPE_VALUE),
                control.message,
                *encode_text(item.arguments),
                control.call,
            )
        )

    def _render_item(self, item: CanonicalItem) -> Message | _GeneratedToolCall:
        if isinstance(item, CanonicalMessage):
            message = Message.from_role_and_content(_ROLES[item.role], item.text)
            if item.role is Role.ASSISTANT:
                # A replayed assistant answer belongs to `final`, the channel it
                # was generated on. Harmony's system block states that a channel
                # must be included for every message, and rendering one without
                # produces a turn the model was never trained on -- and one that
                # no longer matches the tokens it originally produced, so prefix
                # reuse breaks on the next turn too.
                message = message.with_channel(FINAL)
            return message

        if isinstance(item, ReasoningTrace):
            # Replayed chain-of-thought goes back into `analysis`, the channel it
            # came from. Put anywhere else it would read as visible output.
            return Message.from_role_and_content(HarmonyRole.ASSISTANT, item.text).with_channel(
                ANALYSIS
            )

        if isinstance(item, ToolCall):
            return self._render_tool_call(item)

        if isinstance(item, ToolOutput):
            # The result is authored *by the tool*, not by the user. Attributing
            # it to the user would teach the model that the human ran the
            # command, which changes who it addresses next.
            #
            # The namespace has to match the call's. A result from
            # `multi_agent_v1.close_agent` attributed to `functions.close_agent`
            # answers a call the transcript does not contain.
            author = Author.new(
                HarmonyRole.TOOL, recipient_for(item.name or "unknown", item.namespace)
            )
            return Message.from_author_and_content(author, item.output).with_channel(COMMENTARY)

        raise TypeError(f"unrenderable conversation item: {type(item).__name__}")
