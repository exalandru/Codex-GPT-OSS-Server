"""Namespaced tools: declaration, call, result attribution.

The wire facts these tests encode were observed against Codex 0.147, not
inferred:

- Codex declares a namespace as ``{type, name, description, tools}``, and each
  member is an ordinary function tool.
- Codex's router dispatches a call carrying ``name`` and ``namespace`` as
  separate fields, and answers a flattened ``multi_agent_v1.close_agent`` with
  ``unsupported call``.

Rendering goes through the real Harmony encoding. No model weights are loaded.
"""

from __future__ import annotations

import pytest

from quantum_codex.api.schemas import parse_request, to_canonical_turn
from quantum_codex.canonical import (
    CanonicalMessage,
    CanonicalTurn,
    ReasoningEffort,
    Role,
    ToolCall,
    ToolDefinition,
    ToolNamespace,
    ToolOutput,
)
from quantum_codex.codex.capabilities import CAPABILITIES
from quantum_codex.harmony import HarmonyRenderer
from quantum_codex.harmony.parse import split_recipient
from quantum_codex.harmony.render import load_encoding
from quantum_codex.routing import ToolRouter


@pytest.fixture(scope="module")
def renderer() -> HarmonyRenderer:
    return HarmonyRenderer()


def rendered(renderer: HarmonyRenderer, turn: CanonicalTurn) -> str:
    return load_encoding().decode(renderer.render(turn))


SPAWN = ToolDefinition(
    name="spawn_agent",
    description="Start a subagent.",
    parameters={"type": "object", "properties": {"prompt": {"type": "string"}}},
)
SHELL = ToolDefinition(
    name="exec_command",
    description="Run a shell command.",
    parameters={"type": "object", "properties": {"command": {"type": "string"}}},
)
MULTI_AGENT = ToolNamespace(
    name="multi_agent_v1", description="Tools for spawning sub-agents.", tools=(SPAWN,)
)


def turn(**kwargs) -> CanonicalTurn:
    return CanonicalTurn(items=(CanonicalMessage(role=Role.USER, text="go"),), **kwargs)


# -- declaration -------------------------------------------------------------


def test_a_namespace_is_declared_under_its_own_name(renderer: HarmonyRenderer) -> None:
    text = rendered(renderer, turn(tool_namespaces=(MULTI_AGENT,)))

    assert "namespace multi_agent_v1 {" in text
    assert "} // namespace multi_agent_v1" in text
    # Its own description, not the member's.
    assert "// Tools for spawning sub-agents." in text
    assert "type spawn_agent = " in text


def test_a_namespaced_tool_is_not_flattened_into_functions(renderer: HarmonyRenderer) -> None:
    """The discriminating check: the member must not appear as a top-level function.

    A renderer that folded namespaces into `functions` would still show the tool
    and still let the model call it -- and every such call would come back
    unroutable.
    """
    text = rendered(renderer, turn(tool_namespaces=(MULTI_AGENT,)))

    assert "namespace functions {" not in text
    assert "multi_agent_v1_spawn_agent" not in text


def test_functions_and_namespaces_coexist_in_one_turn(renderer: HarmonyRenderer) -> None:
    text = rendered(renderer, turn(tools=(SHELL,), tool_namespaces=(MULTI_AGENT,)))

    assert "namespace functions {" in text
    assert "type exec_command = " in text
    assert "namespace multi_agent_v1 {" in text
    assert "type spawn_agent = " in text


def test_the_rendered_tool_block_is_byte_stable(renderer: HarmonyRenderer) -> None:
    """The rendered tokens are the prompt-cache key.

    An unstable tool block -- set iteration, dict reordering -- would change the
    prefix between two identical turns and silently cost every reuse.
    """
    subject = turn(tools=(SHELL,), tool_namespaces=(MULTI_AGENT,))

    assert renderer.render(subject) == renderer.render(subject)


# -- calls and results -------------------------------------------------------


def test_a_namespaced_call_addresses_its_namespace(renderer: HarmonyRenderer) -> None:
    text = rendered(
        renderer,
        CanonicalTurn(
            items=(
                CanonicalMessage(role=Role.USER, text="go"),
                ToolCall(
                    call_id="c1", name="spawn_agent", arguments="{}", namespace="multi_agent_v1"
                ),
            )
        ),
    )

    assert "to=multi_agent_v1.spawn_agent" in text


def test_a_namespaced_result_is_attributed_to_its_namespace(renderer: HarmonyRenderer) -> None:
    """A result from `functions.spawn_agent` answers a call that never happened."""
    text = rendered(
        renderer,
        CanonicalTurn(
            items=(
                ToolCall(
                    call_id="c1", name="spawn_agent", arguments="{}", namespace="multi_agent_v1"
                ),
                ToolOutput(
                    call_id="c1", output="agent_1", name="spawn_agent", namespace="multi_agent_v1"
                ),
            )
        ),
    )

    assert "<|start|>multi_agent_v1.spawn_agent<|channel|>commentary<|message|>agent_1" in text
    assert "functions.spawn_agent" not in text


def test_an_ordinary_result_stays_in_functions(renderer: HarmonyRenderer) -> None:
    text = rendered(
        renderer,
        CanonicalTurn(
            items=(
                ToolCall(call_id="c1", name="exec_command", arguments="{}"),
                ToolOutput(call_id="c1", output="/tmp", name="exec_command"),
            )
        ),
    )

    assert "functions.exec_command" in text


@pytest.mark.parametrize(
    ("recipient", "expected"),
    [
        ("functions.exec_command", ("exec_command", None)),
        ("multi_agent_v1.spawn_agent", ("spawn_agent", "multi_agent_v1")),
        ("exec_command", ("exec_command", None)),
    ],
)
def test_recipients_split_into_name_and_namespace(
    recipient: str, expected: tuple[str, str | None]
) -> None:
    assert split_recipient(recipient) == expected


# -- capability policy -------------------------------------------------------


def namespace_tool(**overrides) -> dict:
    tool = {
        "type": "namespace",
        "name": "multi_agent_v1",
        "description": "Tools for spawning sub-agents.",
        "tools": [
            {"type": "function", "name": "spawn_agent", "description": "", "parameters": {}}
        ],
    }
    tool.update(overrides)
    return tool


def test_a_namespace_tool_is_kept_not_dropped() -> None:
    plan = CAPABILITIES.plan_tools([namespace_tool()])

    assert plan.problem is None
    assert plan.dropped == ()
    assert [index for index, _ in plan.kept] == [0]


def test_the_functions_namespace_name_is_reserved() -> None:
    """It would collide with top-level tools in Harmony's namespace map."""
    plan = CAPABILITIES.plan_tools([namespace_tool(name="functions")])

    assert plan.problem is not None
    assert plan.problem.param == "tools[0].name"


def test_a_duplicate_namespace_is_refused() -> None:
    plan = CAPABILITIES.plan_tools([namespace_tool(), namespace_tool()])

    assert plan.problem is not None
    assert plan.problem.param == "tools[1].name"


def test_a_non_function_inside_a_namespace_is_refused() -> None:
    plan = CAPABILITIES.plan_tools([namespace_tool(tools=[{"type": "web_search"}])])

    assert plan.problem is not None
    assert plan.problem.param == "tools[0].tools[0].type"


def test_a_namespace_without_a_tools_array_is_refused() -> None:
    tool = namespace_tool()
    del tool["tools"]
    plan = CAPABILITIES.plan_tools([tool])

    assert plan.problem is not None
    assert plan.problem.param == "tools[0].tools"


def test_web_search_is_still_dropped() -> None:
    """Supporting namespaces must not have widened anything else."""
    plan = CAPABILITIES.plan_tools([{"type": "web_search"}])

    assert plan.problem is None
    assert [tool.type for tool in plan.dropped] == ["web_search"]


# -- request normalisation ---------------------------------------------------


def canonical(**body) -> CanonicalTurn:
    request = parse_request({"model": "gpt-oss-20b", "input": "go", **body})
    return to_canonical_turn(request, default_effort=ReasoningEffort.MEDIUM)


def test_a_namespace_tool_becomes_a_tool_namespace() -> None:
    turn = canonical(tools=[namespace_tool()])

    assert turn.tools == ()
    assert len(turn.tool_namespaces) == 1
    namespace = turn.tool_namespaces[0]
    assert namespace.name == "multi_agent_v1"
    assert namespace.description == "Tools for spawning sub-agents."
    assert [tool.name for tool in namespace.tools] == ["spawn_agent"]


def test_top_level_functions_stay_out_of_namespaces() -> None:
    turn = canonical(
        tools=[
            {"type": "function", "name": "exec_command", "parameters": {}},
            namespace_tool(),
        ]
    )

    assert [tool.name for tool in turn.tools] == ["exec_command"]
    assert [ns.name for ns in turn.tool_namespaces] == ["multi_agent_v1"]


def test_a_result_inherits_the_namespace_of_its_call() -> None:
    """The wire names only `call_id`; the namespace has to come from the call."""
    turn = canonical(
        input=[
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "spawn_agent",
                "namespace": "multi_agent_v1",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "c1", "output": "agent_1"},
        ]
    )

    output = turn.items[-1]
    assert isinstance(output, ToolOutput)
    assert output.name == "spawn_agent"
    assert output.namespace == "multi_agent_v1"


def test_an_ordinary_result_inherits_no_namespace() -> None:
    turn = canonical(
        input=[
            {"type": "function_call", "call_id": "c1", "name": "exec_command", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "/tmp"},
        ]
    )

    output = turn.items[-1]
    assert isinstance(output, ToolOutput)
    assert output.namespace is None


# -- routing authority -------------------------------------------------------


def router_for(**kwargs) -> ToolRouter:
    return ToolRouter(turn(**kwargs))


def test_an_explicit_namespace_is_kept() -> None:
    """What the 120B emits. Nothing to repair."""
    router = router_for(tools=(SHELL,), tool_namespaces=(MULTI_AGENT,))

    assert router.resolve("spawn_agent", "multi_agent_v1") == ("spawn_agent", "multi_agent_v1")


def test_a_functions_addressed_namespaced_tool_is_routed_to_its_namespace() -> None:
    """What the 20B emits: `functions.spawn_agent`, which Codex rejects.

    The declaration is authoritative -- `spawn_agent` is not a top-level
    function, and exactly one namespace declares it.
    """
    router = router_for(tools=(SHELL,), tool_namespaces=(MULTI_AGENT,))

    assert router.resolve("spawn_agent", None) == ("spawn_agent", "multi_agent_v1")


def test_a_real_top_level_function_is_left_alone() -> None:
    """The discriminating case: resolution must not invent a namespace."""
    router = router_for(tools=(SHELL,), tool_namespaces=(MULTI_AGENT,))

    assert router.resolve("exec_command", None) == ("exec_command", None)


def test_a_name_declared_in_two_namespaces_is_never_guessed() -> None:
    other = ToolNamespace(name="mcp__srv", tools=(SPAWN,))
    router = router_for(tool_namespaces=(MULTI_AGENT, other))

    assert router.resolve("spawn_agent", None) == ("spawn_agent", None)


def test_a_name_shadowed_by_a_top_level_function_stays_top_level() -> None:
    """A declared function wins: the model addressed `functions` and meant it."""
    shadow = ToolNamespace(name="mcp__srv", tools=(SHELL,))
    router = router_for(tools=(SHELL,), tool_namespaces=(shadow,))

    assert router.resolve("exec_command", None) == ("exec_command", None)


def test_an_undeclared_namespace_is_forwarded_unchanged() -> None:
    """Rewriting it would hide a real prompt or model problem."""
    router = router_for(tool_namespaces=(MULTI_AGENT,))

    assert router.resolve("whatever", "not_declared") == ("whatever", "not_declared")


def test_an_unknown_tool_is_forwarded_unchanged() -> None:
    router = router_for(tools=(SHELL,), tool_namespaces=(MULTI_AGENT,))

    assert router.resolve("mystery", None) == ("mystery", None)


def test_a_functions_prefixed_namespace_is_stripped() -> None:
    """Observed on the 120B: `functions.mcp__witness.reverse_text`."""
    mcp = ToolNamespace(name="mcp__witness", tools=(ToolDefinition(name="reverse_text"),))
    router = ToolRouter(turn(tools=(SHELL,), tool_namespaces=(mcp,)))

    assert router.resolve("reverse_text", "functions.mcp__witness") == (
        "reverse_text",
        "mcp__witness",
    )


def test_a_glued_recipient_is_split_against_the_declarations() -> None:
    """Observed on both models: the separator is dropped entirely."""
    mcp = ToolNamespace(name="mcp__witness", tools=(ToolDefinition(name="reverse_text"),))
    router = ToolRouter(turn(tool_namespaces=(MULTI_AGENT, mcp)))

    assert router.resolve("mcp__witnessreverse_text", None) == ("reverse_text", "mcp__witness")
    assert router.resolve("multi_agent_v1spawn_agent", None) == ("spawn_agent", "multi_agent_v1")


def test_a_prefix_match_alone_does_not_produce_a_route() -> None:
    """The remainder must also be a declared member of that same namespace."""
    router = ToolRouter(turn(tool_namespaces=(MULTI_AGENT,)))

    assert router.resolve("multi_agent_v1_not_a_tool", None) == ("multi_agent_v1_not_a_tool", None)


def test_an_unresolvable_namespace_is_still_forwarded_unchanged() -> None:
    router = ToolRouter(turn(tool_namespaces=(MULTI_AGENT,)))

    assert router.resolve("x", "functions.nope") == ("x", "functions.nope")


def test_a_functions_prefix_is_not_stripped_for_an_undeclared_tool() -> None:
    """Correction requires the namespace to really declare the name.

    Stripping on the prefix alone would route `functions.multi_agent_v1.bogus`
    into a namespace with no such tool -- inventing a route rather than
    normalising one.
    """
    router = ToolRouter(turn(tool_namespaces=(MULTI_AGENT,)))

    assert router.resolve("bogus", "functions.multi_agent_v1") == (
        "bogus",
        "functions.multi_agent_v1",
    )


def test_an_existing_route_is_never_rewritten() -> None:
    """A tool that really is a top-level function stays one, even when a
    namespace declares the same name."""
    shadow = ToolNamespace(name="mcp__srv", tools=(SHELL,))
    router = ToolRouter(turn(tools=(SHELL,), tool_namespaces=(shadow,)))

    assert router.resolve("exec_command", None) == ("exec_command", None)
