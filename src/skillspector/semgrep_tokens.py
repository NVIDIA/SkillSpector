# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded ownership of data fields in supported Semgrep rule documents."""

from __future__ import annotations

from collections.abc import Callable

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from skillspector.yaml_tokens import yaml_source_spans

_STRING = "tag:yaml.org,2002:str"
_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL", "ERROR", "WARNING", "INFO"}
_LANGUAGES = {
    "regex",
    "generic",
    "bash",
    "sh",
    "python",
    "python2",
    "python3",
    "py",
    "javascript",
    "js",
    "typescript",
    "ts",
}
_LEAVES = {
    "pattern",
    "pattern-not",
    "pattern-inside",
    "pattern-not-inside",
    "pattern-regex",
    "pattern-not-regex",
}
_GROUPS = {"patterns", "pattern-either"}
_SELECTORS = {"pattern", "patterns", "pattern-either", "pattern-regex"}
_OPTIONAL = {"metadata", "options", "paths", "min-version", "max-version", "fix"}


def _mapping(node: Node | None) -> dict[str, Node] | None:
    if not isinstance(node, MappingNode):
        return None
    result: dict[str, Node] = {}
    for key, value in node.value:
        if not isinstance(key, ScalarNode) or key.tag != _STRING or key.value in result:
            return None
        result[key.value] = value
    return result


def _text(node: Node | None) -> bool:
    return isinstance(node, ScalarNode) and node.tag == _STRING and bool(node.value)


def semgrep_rule_scalar_roles(content: str, check_runtime: Callable[[], None]) -> dict[int, str]:
    """Prove selected query/text roles, never security finding exemptions.

    The full YAML grammar bounds source size, nesting and token count before
    composing syntax nodes. No constructors or regular expressions execute.
    Unsupported schemas retain the caller's conservative scalar interpretation.
    """
    spans = yaml_source_spans(content, check_runtime)
    if spans is None or not any(scalar.value == "rules" for scalar in spans.scalars):
        return {}
    check_runtime()
    try:
        root = _mapping(yaml.compose(content, Loader=yaml.SafeLoader))
    except (yaml.YAMLError, ValueError, RecursionError):
        return {}
    check_runtime()
    if root is None or set(root) != {"rules"}:
        return {}
    rules = root["rules"]
    if not isinstance(rules, SequenceNode) or not rules.value:
        return {}
    roles: dict[int, str] = {}
    rule_ids: set[str] = set()

    def operator(name: str, value: Node) -> bool:
        check_runtime()
        if name in _LEAVES and _text(value):
            roles[value.start_mark.index] = "pattern"
            return True
        return (
            name in _GROUPS
            and isinstance(value, SequenceNode)
            and bool(value.value)
            and all(formula(child) for child in value.value)
        )

    def formula(node: Node) -> bool:
        fields = _mapping(node)
        return fields is not None and len(fields) == 1 and operator(*next(iter(fields.items())))

    for node in rules.value:
        check_runtime()
        rule = _mapping(node)
        if rule is None or not {"id", "message", "severity", "languages"} <= rule.keys():
            return {}
        if not all(_text(rule[key]) for key in ("id", "message", "severity")):
            return {}
        if rule["id"].value in rule_ids:
            return {}
        rule_ids.add(rule["id"].value)
        if rule["severity"].value not in _SEVERITIES:
            return {}
        languages = rule["languages"]
        if (
            not isinstance(languages, SequenceNode)
            or not languages.value
            or not all(
                _text(language) and language.value in _LANGUAGES for language in languages.value
            )
        ):
            return {}
        allowed = {"id", "message", "severity", "languages"} | _OPTIONAL | _SELECTORS
        if "mode" in rule:
            if not _text(rule["mode"]) or rule["mode"].value != "taint":
                return {}
            allowed |= {"mode", "pattern-sources", "pattern-sinks", "pattern-sanitizers"}
            if not {"pattern-sources", "pattern-sinks"} <= rule.keys() or _SELECTORS & rule.keys():
                return {}
            for name in ("pattern-sources", "pattern-sinks", "pattern-sanitizers"):
                if name not in rule:
                    continue
                values = rule[name]
                if (
                    not isinstance(values, SequenceNode)
                    or not values.value
                    or not all(formula(value) for value in values.value)
                ):
                    return {}
        else:
            selectors = _SELECTORS & rule.keys()
            if len(selectors) != 1:
                return {}
            name = next(iter(selectors))
            if not operator(name, rule[name]):
                return {}
        if not rule.keys() <= allowed:
            return {}
        for name in ("metadata", "options", "paths"):
            if name in rule and _mapping(rule[name]) is None:
                return {}
        for name in ("min-version", "max-version", "fix"):
            if name in rule and not _text(rule[name]):
                return {}
        roles[rule["message"].start_mark.index] = "message"
    return roles
