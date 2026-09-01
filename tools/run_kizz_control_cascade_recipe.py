#!/usr/bin/env python3
"""Plan, preflight, resume, and execute the Kizz Control cascade recipe.

The recipe is a data pipeline, not a shell script. Commands are argv arrays and
are never evaluated by a shell. Each completed stage records the command and
SHA-256 of its declared evidence files, so a changed recipe or drifted output
cannot be silently skipped on resume.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
PROTECTED_RESOURCES = {"hardware", "paid_api"}
SELECTION_ROLES = {"model", "threshold"}


class RecipeError(ValueError):
    """Raised when a recipe or execution boundary is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Stage:
    id: str
    phase: str
    description: str
    depends_on: tuple[str, ...]
    command: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    evidence: tuple[str, ...]
    resources: frozenset[str]
    reads_splits: frozenset[str]
    post_selection_reads_splits: frozenset[str]
    selection_role: str | None


@dataclass(frozen=True)
class Recipe:
    path: Path
    raw: Mapping[str, Any]
    stages: tuple[Stage, ...]
    variables: Mapping[str, Mapping[str, Any]]
    digest: str

    @property
    def by_id(self) -> dict[str, Stage]:
        return {stage.id: stage for stage in self.stages}


def _as_string_tuple(value: Any, field: str, stage_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RecipeError(f"stage {stage_id}: {field} must be a list of strings")
    return tuple(value)


def _placeholders(value: str) -> set[str]:
    names: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(value):
        if field_name:
            names.add(field_name)
    return names


def load_recipe(path: Path) -> Recipe:
    path = path.expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RecipeError("recipe root must be a mapping")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RecipeError(f"recipe schema_version must be {SCHEMA_VERSION}")
    variables = payload.get("variables", {})
    if not isinstance(variables, dict):
        raise RecipeError("variables must be a mapping")
    allowed_variables = set(variables) | {"repo_root"}
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise RecipeError("stages must be a non-empty list")

    stages: list[Stage] = []
    seen: set[str] = set()
    for raw in raw_stages:
        if not isinstance(raw, dict):
            raise RecipeError("each stage must be a mapping")
        stage_id = raw.get("id")
        if not isinstance(stage_id, str) or not stage_id:
            raise RecipeError("each stage needs a non-empty string id")
        if stage_id in seen:
            raise RecipeError(f"duplicate stage id: {stage_id}")
        seen.add(stage_id)
        command = _as_string_tuple(raw.get("command"), "command", stage_id)
        if not command:
            raise RecipeError(f"stage {stage_id}: command cannot be empty")
        stage = Stage(
            id=stage_id,
            phase=str(raw.get("phase", "unspecified")),
            description=str(raw.get("description", "")),
            depends_on=_as_string_tuple(raw.get("depends_on"), "depends_on", stage_id),
            command=command,
            inputs=_as_string_tuple(raw.get("inputs"), "inputs", stage_id),
            outputs=_as_string_tuple(raw.get("outputs"), "outputs", stage_id),
            evidence=_as_string_tuple(raw.get("evidence"), "evidence", stage_id),
            resources=frozenset(_as_string_tuple(raw.get("resources"), "resources", stage_id)),
            reads_splits=frozenset(_as_string_tuple(raw.get("reads_splits"), "reads_splits", stage_id)),
            post_selection_reads_splits=frozenset(
                _as_string_tuple(
                    raw.get("post_selection_reads_splits"),
                    "post_selection_reads_splits",
                    stage_id,
                )
            ),
            selection_role=raw.get("selection_role"),
        )
        if stage.selection_role is not None and stage.selection_role not in SELECTION_ROLES:
            raise RecipeError(
                f"stage {stage_id}: selection_role must be one of {sorted(SELECTION_ROLES)}"
            )
        if stage.selection_role and "test" in stage.reads_splits:
            raise RecipeError(
                f"stage {stage_id}: {stage.selection_role} selection cannot read test"
            )
        used = set()
        for value in (*stage.command, *stage.inputs, *stage.outputs, *stage.evidence):
            used.update(_placeholders(value))
        unknown = used - allowed_variables
        if unknown:
            raise RecipeError(
                f"stage {stage_id}: unknown variables {', '.join(sorted(unknown))}"
            )
        stages.append(stage)

    by_id = {stage.id: stage for stage in stages}
    for stage in stages:
        missing = set(stage.depends_on) - set(by_id)
        if missing:
            raise RecipeError(
                f"stage {stage.id}: unknown dependencies {', '.join(sorted(missing))}"
            )
        if stage.id in stage.depends_on:
            raise RecipeError(f"stage {stage.id}: cannot depend on itself")
    topological_order(tuple(stages))

    return Recipe(
        path=path,
        raw=payload,
        stages=tuple(stages),
        variables=variables,
        digest=canonical_sha256(payload),
    )


def topological_order(stages: tuple[Stage, ...]) -> list[str]:
    remaining = {stage.id: set(stage.depends_on) for stage in stages}
    order: list[str] = []
    while remaining:
        ready = sorted(stage_id for stage_id, deps in remaining.items() if not deps)
        if not ready:
            raise RecipeError(
                "stage dependency cycle: " + ", ".join(sorted(remaining))
            )
        order.extend(ready)
        for stage_id in ready:
            remaining.pop(stage_id)
        for deps in remaining.values():
            deps.difference_update(ready)
    return order


def parse_overrides(values: Iterable[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise RecipeError(f"--set requires NAME=VALUE, got {value!r}")
        name, resolved = value.split("=", 1)
        if not name:
            raise RecipeError("--set variable name cannot be empty")
        overrides[name] = resolved
    return overrides


def resolve_variables(
    recipe: Recipe,
    overrides: Mapping[str, str],
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = os.environ if environment is None else environment
    unknown = set(overrides) - set(recipe.variables)
    if unknown:
        raise RecipeError(f"unknown --set variables: {', '.join(sorted(unknown))}")
    values: dict[str, str] = {
        "repo_root": str(recipe.path.parents[2]),
    }
    missing: list[str] = []
    for name, spec in recipe.variables.items():
        if not isinstance(spec, dict):
            raise RecipeError(f"variable {name}: definition must be a mapping")
        value: Any = overrides.get(name)
        if value is None and spec.get("env"):
            value = environment.get(str(spec["env"]))
        if value is None and "default" in spec:
            value = spec["default"]
        if value in (None, ""):
            if spec.get("required", False):
                missing.append(name)
            continue
        values[name] = str(value)
    if missing:
        raise RecipeError(
            "missing required variables: " + ", ".join(sorted(missing))
        )
    # Defaults may be expressed relative to another variable (normally the
    # workspace). Resolve in bounded passes and fail rather than leaving a
    # literal placeholder in a command.
    for _ in range(len(values) + 1):
        changed = False
        for name, value in tuple(values.items()):
            try:
                resolved = value.format_map(values)
            except KeyError:
                continue
            resolved = os.path.expanduser(resolved)
            if resolved != value:
                values[name] = resolved
                changed = True
        if not changed:
            break
    unresolved = {
        placeholder
        for value in values.values()
        for placeholder in _placeholders(value)
    }
    if unresolved:
        raise RecipeError(
            "unresolved variable defaults: " + ", ".join(sorted(unresolved))
        )
    return values


def render(value: str, variables: Mapping[str, str]) -> str:
    try:
        return value.format_map(variables)
    except KeyError as error:
        raise RecipeError(f"unresolved recipe variable: {error.args[0]}") from error


def render_command_token(value: str, variables: Mapping[str, str]) -> str:
    """Render one argv token, resolving an explicit provenance hash token."""
    resolved = render(value, variables)
    prefix = "@sha256:"
    if not resolved.startswith(prefix):
        return resolved
    path = Path(resolved[len(prefix) :]).expanduser().resolve()
    if not path.is_file():
        raise RecipeError(f"cannot hash missing command input: {path}")
    return sha256_file(path)


def dependency_closure(stage_ids: Iterable[str], by_id: Mapping[str, Stage]) -> set[str]:
    selected: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id not in by_id:
            raise RecipeError(f"unknown stage: {stage_id}")
        if stage_id in selected:
            return
        selected.add(stage_id)
        for dependency in by_id[stage_id].depends_on:
            visit(dependency)

    for stage_id in stage_ids:
        visit(stage_id)
    return selected


def selected_stage_ids(recipe: Recipe, requested: Iterable[str]) -> set[str]:
    requested = tuple(requested)
    if not requested:
        return {stage.id for stage in recipe.stages}
    return dependency_closure(requested, recipe.by_id)


def _existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def preflight(
    recipe: Recipe,
    variables: Mapping[str, str],
    selected: set[str],
    *,
    allow_paid: bool,
    allow_hardware: bool,
) -> list[str]:
    errors: list[str] = []
    stages = [stage for stage in recipe.stages if stage.id in selected]
    if not allow_paid:
        paid = [stage.id for stage in stages if "paid_api" in stage.resources]
        if paid:
            errors.append("paid API stages require --allow-paid: " + ", ".join(paid))
    if not allow_hardware:
        hardware = [stage.id for stage in stages if "hardware" in stage.resources]
        if hardware:
            errors.append(
                "hardware stages require --allow-hardware: " + ", ".join(hardware)
            )

    rendered_outputs = {
        str(Path(render(output, variables)).expanduser().resolve())
        for stage in stages
        for output in stage.outputs
    }
    for stage in stages:
        command = [render(part, variables) for part in stage.command]
        executable = command[0]
        if "/" in executable:
            if not Path(executable).expanduser().exists():
                errors.append(f"{stage.id}: executable does not exist: {executable}")
        elif shutil.which(executable) is None:
            errors.append(f"{stage.id}: executable not found on PATH: {executable}")
        for raw_input in stage.inputs:
            input_path = Path(render(raw_input, variables)).expanduser().resolve()
            if str(input_path) not in rendered_outputs and not input_path.exists():
                errors.append(f"{stage.id}: missing external input: {input_path}")

    minimum_free_gb = float(recipe.raw.get("minimum_free_gb", 0))
    if minimum_free_gb:
        workspace = Path(variables.get("workspace", recipe.path.parent))
        free = shutil.disk_usage(_existing_parent(workspace)).free / (1024**3)
        if free < minimum_free_gb:
            errors.append(
                f"workspace has {free:.1f} GiB free; recipe requires {minimum_free_gb:.1f} GiB"
            )
    return errors


def _state_path(variables: Mapping[str, str]) -> Path:
    if "workspace" not in variables:
        raise RecipeError("recipe must define a workspace variable")
    return Path(variables["workspace"]).expanduser().resolve() / "recipe-state.json"


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": STATE_SCHEMA_VERSION, "stages": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RecipeError(f"unsupported state schema in {path}")
    if not isinstance(state.get("stages"), dict):
        raise RecipeError(f"invalid stage state in {path}")
    return state


def write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def stage_signature(stage: Stage, variables: Mapping[str, str]) -> str:
    return canonical_sha256(
        {
            "command": [render(part, variables) for part in stage.command],
            "inputs": [render(path, variables) for path in stage.inputs],
            "outputs": [render(path, variables) for path in stage.outputs],
            "evidence": [render(path, variables) for path in stage.evidence],
        }
    )


def evidence_hashes(stage: Stage, variables: Mapping[str, str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw in stage.evidence:
        path = Path(render(raw, variables)).expanduser().resolve()
        if not path.is_file():
            raise RecipeError(f"stage {stage.id}: evidence file missing: {path}")
        hashes[str(path)] = sha256_file(path)
    return hashes


def stage_is_current(
    stage: Stage,
    variables: Mapping[str, str],
    recipe: Recipe,
    state: Mapping[str, Any],
) -> bool:
    record = state.get("stages", {}).get(stage.id)
    if not isinstance(record, dict):
        return False
    if record.get("recipe_sha256") != recipe.digest:
        return False
    if record.get("stage_sha256") != stage_signature(stage, variables):
        return False
    for raw in stage.outputs:
        if not Path(render(raw, variables)).expanduser().exists():
            return False
    recorded = record.get("evidence_sha256")
    if not isinstance(recorded, dict):
        return False
    try:
        return evidence_hashes(stage, variables) == recorded
    except RecipeError:
        return False


def _run_one_stage(
    stage: Stage,
    variables: Mapping[str, str],
    repo_root: Path,
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    command = [render_command_token(part, variables) for part in stage.command]
    print(f"[{stage.id}] {' '.join(command)}", flush=True)
    if dry_run:
        return None
    for raw in stage.inputs:
        path = Path(render(raw, variables)).expanduser()
        if not path.exists():
            raise RecipeError(f"stage {stage.id}: input missing at execution: {path}")
    result = subprocess.run(command, cwd=repo_root, check=False)
    if result.returncode:
        raise RecipeError(f"stage {stage.id}: command exited {result.returncode}")
    for raw in stage.outputs:
        path = Path(render(raw, variables)).expanduser()
        if not path.exists():
            raise RecipeError(f"stage {stage.id}: declared output missing: {path}")
    return {"evidence_sha256": evidence_hashes(stage, variables)}


def execute(
    recipe: Recipe,
    variables: Mapping[str, str],
    selected: set[str],
    *,
    jobs: int,
    dry_run: bool,
    force: bool,
) -> None:
    if jobs < 1:
        raise RecipeError("--jobs must be at least 1")
    by_id = recipe.by_id
    state_path = _state_path(variables)
    state = load_state(state_path)
    pending = set(selected)
    completed: set[str] = set()
    repo_root = Path(variables["repo_root"])
    while pending:
        ready = sorted(
            stage_id
            for stage_id in pending
            if set(by_id[stage_id].depends_on).issubset(completed | (set(by_id) - selected))
        )
        if not ready:
            raise RecipeError("selected stage graph made no progress")
        runnable: list[Stage] = []
        for stage_id in ready:
            stage = by_id[stage_id]
            if not force and stage_is_current(stage, variables, recipe, state):
                print(f"[{stage.id}] current; skipping", flush=True)
                completed.add(stage.id)
                pending.remove(stage.id)
            else:
                runnable.append(stage)
        if not runnable:
            continue
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    _run_one_stage,
                    stage,
                    variables,
                    repo_root,
                    dry_run=dry_run,
                ): stage
                for stage in runnable
            }
            results: list[tuple[Stage, dict[str, Any] | None]] = []
            for future in concurrent.futures.as_completed(futures):
                stage = futures[future]
                results.append((stage, future.result()))
        for stage, result in sorted(results, key=lambda item: item[0].id):
            if result is not None:
                state.setdefault("stages", {})[stage.id] = {
                    "recipe_sha256": recipe.digest,
                    "stage_sha256": stage_signature(stage, variables),
                    **result,
                }
                write_state(state_path, state)
            completed.add(stage.id)
            pending.remove(stage.id)


def print_plan(recipe: Recipe, variables: Mapping[str, str], selected: set[str]) -> None:
    state = load_state(_state_path(variables))
    order = topological_order(recipe.stages)
    for stage_id in order:
        if stage_id not in selected:
            continue
        stage = recipe.by_id[stage_id]
        status = "current" if stage_is_current(stage, variables, recipe, state) else "pending"
        resources = f" [{', '.join(sorted(stage.resources))}]" if stage.resources else ""
        print(f"{status:7} {stage.phase:16} {stage.id}{resources}")
        if stage.description:
            print(f"         {stage.description}")


def build_parser() -> argparse.ArgumentParser:
    default_recipe = Path(__file__).resolve().parents[1] / "recipes/kizz/control-cascade-v10.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "run", "status"))
    parser.add_argument("--recipe", type=Path, default=default_recipe)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--stage", action="append", default=[], help="run/inspect this stage and its dependencies")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--allow-paid", action="store_true", help="authorize selected paid synthesis API calls")
    parser.add_argument("--allow-hardware", action="store_true", help="authorize selected physical capture stages")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        recipe = load_recipe(args.recipe)
        variables = resolve_variables(recipe, parse_overrides(args.set))
        selected = selected_stage_ids(recipe, args.stage)
        if args.action in {"plan", "status"}:
            print_plan(recipe, variables, selected)
            return 0
        errors = preflight(
            recipe,
            variables,
            selected,
            allow_paid=args.allow_paid,
            allow_hardware=args.allow_hardware,
        )
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2
        print("preflight passed")
        if args.action == "preflight":
            return 0
        execute(
            recipe,
            variables,
            selected,
            jobs=args.jobs,
            dry_run=args.dry_run,
            force=args.force,
        )
        return 0
    except (OSError, RecipeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
