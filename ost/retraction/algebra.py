"""Retraction algebra over the provenance graph.

Reliability scores start at one and only ever decrease. A refutation lowers the refuted
claim's own score; propagation bounds every descendant by its least reliable ancestor.
Taking a minimum rather than a product avoids depth-dependent decay, so a long chain of
sound reasoning is not attenuated merely for being long.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple


class ProvenanceError(ValueError):
    """Raised when the provenance graph is not usable for propagation."""


def score_ceiling(margin: float, rho_min: float, kappa: float) -> float:
    """Refutation ceiling ``alpha(gamma)``.

    Implements App. A.4: ``alpha(gamma) = max(rho_min, exp(-kappa * gamma))``. The
    ceiling decreases with the contradiction margin and never falls below the floor.
    """
    if not 0.0 < rho_min < 1.0:
        raise ValueError("rho_min must lie in (0, 1)")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    clamped = min(max(float(margin), 0.0), 1.0)
    return max(rho_min, math.exp(-kappa * clamped))


def apply_refutation(
    reliability: float, margin: float, rho_min: float, kappa: float
) -> float:
    """Update a refuted claim's own reliability score.

    Implements Eq. (5): ``rho_i <- min(rho_i, alpha(gamma_i))``. Because the update is a
    minimum, reliability is monotone non-increasing, which Proposition 1 relies on.
    """
    return min(float(reliability), score_ceiling(margin, rho_min, kappa))


def topological_order(parents: Mapping[int, Sequence[int]]) -> List[int]:
    """Parent-before-child order over the provenance graph.

    Support citations always point to earlier records, so the graph is acyclic
    (App. A.2). A cycle means the citation guard was bypassed and is an error rather
    than something to silently work around.
    """
    indegree: Dict[int, int] = {node: 0 for node in parents}
    children: Dict[int, List[int]] = {node: [] for node in parents}
    for node, node_parents in parents.items():
        for parent in node_parents:
            if parent not in parents:
                # A parent outside the active subgraph contributes its bound through
                # ``ancestor_bounds`` but does not participate in ordering.
                continue
            indegree[node] += 1
            children[parent].append(node)

    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order: List[int] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort()

    if len(order) != len(parents):
        remaining = sorted(set(parents) - set(order))
        raise ProvenanceError(
            f"provenance graph contains a cycle among spans {remaining}; support "
            "citations must reference earlier records only"
        )
    return order


def propagate_effective(
    reliability: Mapping[int, float],
    parents: Mapping[int, Sequence[int]],
    *,
    external_bounds: Optional[Mapping[int, float]] = None,
) -> Dict[int, float]:
    """Propagate effective reliability over the provenance graph.

    Implements Eq. (5) via the App. A.4 parent-before-child scan::

        a_j       = min_{i in pa(j)} min(rho_i, a_i)     (a_j = 1 for an empty parent set)
        rho_bar_j = rho_j * a_j

    ``a_j`` is the ancestor bound, so ``rho_bar_j = rho_j * min_{k in Anc(j)} rho_k``
    without multiplying every ancestor's weight. ``external_bounds`` supplies bounds for
    parents that have already left the active subgraph.
    """
    order = topological_order(parents)
    bounds: Dict[int, float] = {}
    effective: Dict[int, float] = {}
    external = dict(external_bounds or {})

    for node in order:
        own = float(reliability.get(node, 1.0))
        bound = 1.0
        for parent in parents.get(node, ()):
            if parent in bounds:
                parent_own = float(reliability.get(parent, 1.0))
                bound = min(bound, parent_own, bounds[parent])
            elif parent in external:
                bound = min(bound, float(external[parent]))
            elif parent in reliability:
                bound = min(bound, float(reliability[parent]))
            # A parent that is entirely unknown contributes no bound; the caller is
            # responsible for supplying external bounds for archived ancestors.
        bounds[node] = bound
        effective[node] = own * bound
    return effective


def ancestors(parents: Mapping[int, Sequence[int]], node: int) -> Set[int]:
    """Transitive support ancestors ``Anc(node)``. Revision links are excluded."""
    seen: Set[int] = set()
    stack = list(parents.get(node, ()))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(parents.get(current, ()))
    return seen


def descendants(parents: Mapping[int, Sequence[int]], node: int) -> Set[int]:
    """Transitive descendants of ``node`` in the support graph."""
    children: Dict[int, List[int]] = {}
    for child, child_parents in parents.items():
        for parent in child_parents:
            children.setdefault(parent, []).append(child)
    seen: Set[int] = set()
    stack = list(children.get(node, ()))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(children.get(current, ()))
    return seen


def verify_non_amplification(
    reliability: Mapping[int, float],
    parents: Mapping[int, Sequence[int]],
    effective: Mapping[int, float],
    *,
    tolerance: float = 1e-9,
) -> List[Tuple[int, int]]:
    """Check Proposition 1 and return the violating ``(span, ancestor)`` pairs.

    Proposition 1 states ``rho_bar_j <= rho_i`` for every registered ancestor ``i`` of
    ``j``: a retraction can never leave a dependent span more reliable than what it
    depends on. An empty result is the assertion that propagation held.
    """
    violations: List[Tuple[int, int]] = []
    for node in parents:
        node_effective = float(effective.get(node, 1.0))
        for ancestor in ancestors(parents, node):
            ancestor_own = float(reliability.get(ancestor, 1.0))
            if node_effective > ancestor_own + tolerance:
                violations.append((node, ancestor))
    return violations


__all__ = [
    "ProvenanceError",
    "ancestors",
    "apply_refutation",
    "descendants",
    "propagate_effective",
    "score_ceiling",
    "topological_order",
    "verify_non_amplification",
]
