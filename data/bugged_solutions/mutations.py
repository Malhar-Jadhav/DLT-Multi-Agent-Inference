"""Source-level bug injection for Python programs.

Implements the two mutation families named in the proposal:
  * operator mutations: swap + and -, swap < and <=, swap > and >=, swap == and !=
  * off-by-one mutations: add 1 to an integer literal

Mutations work on the abstract syntax tree. A "site" is one place in the program where a
mutation can be applied, and apply_mutation changes exactly one site, so each mutant differs
from the original by a single edit. Note that ast.unparse drops comments, which is acceptable
because critics see the code, not the original formatting.
"""

import ast
from dataclasses import dataclass
from typing import List

_SWAPS = {
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}


@dataclass
class Site:
    """One mutable location: the AST node, the kind of mutation, and where to find it."""

    kind: str  # "operator" or "off_by_one"
    node: ast.AST
    index: int = 0  # position inside node.ops for comparisons
    line: int = 0


def find_sites(tree: ast.AST) -> List[Site]:
    """Collect every place in the tree where one of the supported mutations can be applied."""
    sites: List[Site] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.BinOp) and type(node.op) in _SWAPS:
            sites.append(Site("operator", node, 0, line))
        elif isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in _SWAPS:
                    sites.append(Site("operator", node, i, line))
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            sites.append(Site("off_by_one", node, 0, line))
    return sites


def apply_mutation(site: Site) -> str:
    """Apply the mutation at a site in place and return a short human readable description."""
    node = site.node
    if site.kind == "off_by_one":
        old = node.value
        node.value = old + 1
        return f"line {site.line}: integer literal {old} changed to {old + 1}"
    if isinstance(node, ast.BinOp):
        old_op = type(node.op)
        node.op = _SWAPS[old_op]()
    else:
        old_op = type(node.ops[site.index])
        node.ops[site.index] = _SWAPS[old_op]()
    return f"line {site.line}: operator {old_op.__name__} changed to {_SWAPS[old_op].__name__}"


def mutate_source(source: str, site_index: int) -> "tuple[str, str, str]":
    """Return (mutated_source, kind, description) after mutating the site_index-th site of source.

    Raises IndexError if the program has fewer sites, and SyntaxError if it does not parse.
    """
    tree = ast.parse(source)
    sites = find_sites(tree)
    site = sites[site_index]
    description = apply_mutation(site)
    return ast.unparse(tree), site.kind, description


def count_sites(source: str) -> int:
    """Return how many mutation sites a program has."""
    return len(find_sites(ast.parse(source)))
