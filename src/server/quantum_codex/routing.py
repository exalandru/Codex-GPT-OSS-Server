"""Normalising a tool call's recipient (D7).

The declared tool topology is authoritative for one narrow thing: the
**unambiguous normalisation** of a recipient the model emitted.

A recipient may be corrected only when both hold:

1. the emitted route **does not exist** -- no declared tool sits at it; and
2. exactly one declared tool provides a **structurally justified** mapping,
   meaning the corrected namespace really declares the corrected name.

Everything else is left exactly as the model produced it: an unknown namespace,
a name nothing declares, and any mapping that more than one declared tool could
satisfy. Those are forwarded unmodified and logged, and the client refuses them.

This is normalisation, not routing authority. The router never chooses between
candidates and never invents a route -- it only removes a mis-spelling when the
declarations leave a single possible reading.

This matters because the stock models behave differently from each other -- the
measurements below are of the 20B and the 120B, and the tuned build is a 120B
fine-tune that has not been measured separately here. Harmony's
system block carries one hardcoded routing sentence, ``Calls to these tools must
go to the commentary channel: 'functions'.``, and there is no way to make it name
another namespace. Measured on the real models with a namespace declared and a
prompt that asks for it (12 samples each, temperature 1.0):

===============================  ==========  ==========
variant                          20B         120B
===============================  ==========  ==========
developer namespace              0/3         2/3
developer namespace + rule       0/3         3/3
system namespace                 0/3         2/3
system namespace + rule          0/3         3/3
===============================  ==========  ==========

So the 120B addresses ``multi_agent_v1.spawn_agent`` correctly, and the 20B
always addresses ``functions.spawn_agent`` -- a call Codex rejects with
``unsupported call``. Resolving the namespace from the declaration makes the
20B usable without pretending it emitted something it did not.

Ambiguity is never guessed. If a name is declared in more than one namespace,
the recipient is left exactly as the model produced it and the client decides,
because inventing a route is how a call reaches the wrong tool.
"""

from __future__ import annotations

import logging

from .canonical import CanonicalTurn
from .harmony.render import FUNCTIONS_NAMESPACE as FUNCTIONS

logger = logging.getLogger(__name__)


class ToolRouter:
    """Resolves a model-emitted recipient against the turn's declarations."""

    def __init__(self, turn: CanonicalTurn) -> None:
        self._functions = {tool.name for tool in turn.tools}
        # name -> namespaces declaring it. A list, not a single value, because
        # the ambiguous case has to stay distinguishable from the resolved one.
        self._by_name: dict[str, list[str]] = {}
        for namespace in turn.tool_namespaces:
            for tool in namespace.tools:
                self._by_name.setdefault(tool.name, []).append(namespace.name)
        self._declared = {namespace.name for namespace in turn.tool_namespaces}
        self._repaired: set[str] = set()

    def resolve(self, name: str, namespace: str | None) -> tuple[str, str | None]:
        """``(name, namespace)`` as it should go on the wire.

        Codex dispatches on ``name`` and ``namespace`` as separate fields and
        rejects a flattened ``ns.name``, so this returns the pair rather than a
        composed string.

        Three mis-addressings have been observed in real sessions, all of them
        the hardcoded ``'functions'`` sentence bleeding into the recipient::

            functions.spawn_agent                  namespace dropped        (20B)
            functions.mcp__witness.reverse_text    `functions.` prefixed    (120B)
            functions.mcp__witnessreverse_text     separator dropped        (120B)

        Each is normalised only when the emitted route does not exist and one
        declared tool justifies the correction structurally. Ambiguous and
        unrecognised recipients are returned untouched.
        """
        if namespace is not None:
            # `functions.<ns>` -- the model reached the right namespace and put
            # the system block's one namespace in front of it. Corrected only
            # when `<ns>` really declares this tool: the prefix alone would
            # route `functions.multi_agent_v1.bogus` to a namespace that has no
            # such tool, inventing a route rather than normalising one.
            head, _, tail = namespace.partition(".")
            if head == FUNCTIONS and tail and tail in self._by_name.get(name, []):
                return self._repair(name, tail, f"{namespace}.{name}")

            if namespace not in self._declared:
                # A namespace nothing declared, and nothing in the declarations
                # resolves it. Passed through unchanged: the client is the
                # router, and silently rewriting a recipient the model was
                # explicit about would hide a real prompt or model problem.
                logger.warning(
                    "tool call addressed to undeclared namespace %r (tool %r); "
                    "forwarding it unchanged",
                    namespace,
                    name,
                )
            return (name, namespace)

        # No namespace: an ordinary function, or a namespaced tool addressed to
        # `functions` because that is the only routing instruction Harmony gives.
        if name in self._functions:
            return (name, None)

        owners = self._by_name.get(name, [])
        if len(owners) == 1:
            return self._repair(name, owners[0], f"functions.{name}")

        if len(owners) > 1:
            logger.warning(
                "tool %r is declared in namespaces %s; the call did not say which, "
                "so it is forwarded unresolved",
                name,
                ", ".join(sorted(owners)),
            )
            return (name, None)

        # `<namespace><tool>` with the separator missing. Only accepted when
        # exactly one declared namespace is a prefix and the remainder is one of
        # its own tools -- two independent facts, so a coincidental prefix match
        # cannot produce a route on its own.
        glued = [
            (member, ns)
            for ns in self._declared
            if name.startswith(ns)
            for member in [name[len(ns) :]]
            if ns in self._by_name.get(member, [])
        ]
        if len(glued) == 1:
            member, owner = glued[0]
            return self._repair(member, owner, name)
        if len(glued) > 1:
            logger.warning(
                "tool call %r splits into more than one declared namespace; "
                "forwarding it unresolved",
                name,
            )

        return (name, None)

    def _repair(self, name: str, namespace: str, emitted: str) -> tuple[str, str | None]:
        """Route a mis-addressed call, announcing it once per distinct recipient."""
        if emitted not in self._repaired:
            self._repaired.add(emitted)
            logger.info(
                "tool call emitted as %r; %r is declared in namespace %r, routing it there",
                emitted,
                name,
                namespace,
            )
        return (name, namespace)
