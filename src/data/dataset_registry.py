"""Dataset and target definitions shared by generation and training code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetSpec:
    unit: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    artifact_stem: str
    image_namespace: str
    task_type: str
    targets: dict[str, TargetSpec]
    include_in_default_evaluation: bool = True


@dataclass(frozen=True)
class ResolvedTask:
    task_name: str
    dataset_name: str
    target_column: str | None


QM9_TARGETS = ("homo", "lumo", "mu")
QM9_TASK_TO_TARGET = {f"QM9_{target}": target for target in QM9_TARGETS}
QM9_TASKS = tuple(QM9_TASK_TO_TARGET)

DATASET_SPECS: dict[str, DatasetSpec] = {
    "QM9": DatasetSpec(
        name="QM9",
        artifact_stem="QM9",
        image_namespace="QM9",
        task_type="regression",
        targets={
            "homo": TargetSpec(unit="Hartree"),
            "lumo": TargetSpec(unit="Hartree"),
            "mu": TargetSpec(unit="Debye"),
        },
        include_in_default_evaluation=False,
    ),
}


def resolve_task_name(task_name: str, allow_dataset_only: bool = False) -> ResolvedTask:
    if task_name in QM9_TASK_TO_TARGET:
        return ResolvedTask(task_name, "QM9", QM9_TASK_TO_TARGET[task_name])
    if task_name == "QM9":
        if not allow_dataset_only:
            choices = ", ".join(QM9_TASKS)
            raise ValueError(f"QM9 requires a task-specific name; choose one of: {choices}")
        return ResolvedTask(task_name, "QM9", None)
    return ResolvedTask(task_name, task_name, task_name)


def get_dataset_spec(task_name: str) -> DatasetSpec | None:
    resolved = resolve_task_name(task_name, allow_dataset_only=True)
    return DATASET_SPECS.get(resolved.dataset_name)


def dataset_name_for_task(task_name: str) -> str:
    return resolve_task_name(task_name, allow_dataset_only=True).dataset_name


def target_columns_for_generation(task_name: str) -> tuple[str, ...]:
    spec = get_dataset_spec(task_name)
    if spec is None:
        return (task_name,)
    return tuple(spec.targets)


def target_column_for_task(task_name: str) -> str:
    resolved = resolve_task_name(task_name)
    if resolved.target_column is None:  # Defensive; resolve_task_name rejects this.
        raise ValueError(f"task {task_name!r} has no target column")
    return resolved.target_column


def task_name_from_dataset_target(dataset_name: str, target_column: str) -> str:
    if dataset_name == "QM9":
        task_name = f"QM9_{target_column}"
        if task_name not in QM9_TASK_TO_TARGET:
            choices = ", ".join(QM9_TASKS)
            raise ValueError(
                f"unsupported QM9 target {target_column!r}; tasks are: {choices}"
            )
        return task_name
    return target_column


def target_unit(task_name: str) -> str | None:
    resolved = resolve_task_name(task_name)
    spec = DATASET_SPECS.get(resolved.dataset_name)
    if spec is None:
        return None
    return spec.targets[resolved.target_column].unit
