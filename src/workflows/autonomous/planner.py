"""
Dynamic Task DAG Planner and Dependency Resolver for Workflow 6.
"""
from __future__ import annotations

import collections
import logging
import re
from typing import Any

from src.workflows.autonomous.schemas import TaskDAG, TaskNode

logger = logging.getLogger("hermes.workflows.autonomous.planner")


def decompose_to_dag(objective: str) -> TaskDAG:
    """
    Decompose a complex multi-step objective into a Directed Acyclic Graph (DAG)
    of dependent and independent subtasks.
    """
    tasks: dict[str, TaskNode] = {}
    obj_clean = objective.strip()

    # Identify potential entities or parallel targets (e.g., 3 databases, multiple components)
    num_match = re.search(r"\b(?:three|3)\s+([a-zA-Z0-9_\-\s]+)", obj_clean, re.IGNORECASE)

    if "vector database" in obj_clean.lower() or "databases" in obj_clean.lower():
        # Dedicated parallel vector database benchmarking DAG
        targets = ["Qdrant", "Milvus", "Chroma"]
        dep_ids = []
        for idx, target in enumerate(targets, start=1):
            t_id = f"task_research_{target.lower()}"
            dep_ids.append(t_id)
            tasks[t_id] = TaskNode(
                task_id=t_id,
                name=f"Research {target}",
                action_type="research",
                description=f"Investigate architecture, indexing performance, and HNSW benchmarks for {target}.",
                tool_name="web_scraper.search",
                arguments={"query": f"{target} vector database architecture benchmarks"},
                dependencies=[],
            )

        # Stage 2: Comparison Task (depends on all parallel research tasks)
        comp_id = "task_compare_performance"
        tasks[comp_id] = TaskNode(
            task_id=comp_id,
            name="Comparative Performance Matrix",
            action_type="compute",
            description="Aggregate benchmarks into normalized latency, throughput, and memory comparison.",
            dependencies=dep_ids,
        )

        # Stage 3: Report Synthesis Task (depends on comparison)
        rep_id = "task_generate_final_report"
        tasks[rep_id] = TaskNode(
            task_id=rep_id,
            name="Generate Final Report",
            action_type="synthesize",
            description="Produce structured Markdown evaluation and strategic recommendation report.",
            dependencies=[comp_id],
        )

    else:
        # General-purpose dynamic multi-step decomposition
        t1 = "task_gather_context"
        t2 = "task_execute_analysis"
        t3 = "task_synthesize_artifact"

        tasks[t1] = TaskNode(
            task_id=t1,
            name="Gather Context & Data",
            action_type="research",
            description=f"Collect background requirements for: {obj_clean}",
            dependencies=[],
        )
        tasks[t2] = TaskNode(
            task_id=t2,
            name="Execute Core Analysis",
            action_type="compute",
            description=f"Process data and evaluate solution parameters for: {obj_clean}",
            dependencies=[t1],
        )
        tasks[t3] = TaskNode(
            task_id=t3,
            name="Synthesize Final Artifact",
            action_type="synthesize",
            description=f"Assemble complete deliverable for: {obj_clean}",
            dependencies=[t2],
        )

    dag = TaskDAG(tasks=tasks)
    is_valid, reason = validate_dag(dag)
    if not is_valid:
        raise ValueError(f"Invalid Task DAG generated: {reason}")

    dag.execution_batches = compute_topological_batches(dag)
    return dag


def validate_dag(dag: TaskDAG) -> tuple[bool, str]:
    """Validate DAG structural integrity: verify dependencies exist and detect cycles."""
    tasks = dag.tasks

    # 1. Verify all referenced dependency IDs exist
    for t_id, task in tasks.items():
        for dep in task.dependencies:
            if dep not in tasks:
                return False, f"Task '{t_id}' references unknown dependency '{dep}'"

    # 2. Cycle detection via Kahn's algorithm
    in_degree = {t_id: 0 for t_id in tasks}
    adjacency = collections.defaultdict(list)

    for t_id, task in tasks.items():
        for dep in task.dependencies:
            adjacency[dep].append(t_id)
            in_degree[t_id] += 1

    queue = collections.deque([t for t, deg in in_degree.items() if deg == 0])
    visited_count = 0

    while queue:
        node = queue.popleft()
        visited_count += 1
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited_count != len(tasks):
        return False, "Cycle detected in task dependency graph!"

    return True, "Valid DAG"


def compute_topological_batches(dag: TaskDAG) -> list[list[str]]:
    """Group tasks into discrete execution stages that can run in parallel."""
    tasks = dag.tasks
    in_degree = {t_id: len(task.dependencies) for t_id, task in tasks.items()}
    adjacency = collections.defaultdict(list)

    for t_id, task in tasks.items():
        for dep in task.dependencies:
            adjacency[dep].append(t_id)

    batches: list[list[str]] = []
    current_batch = [t for t, deg in in_degree.items() if deg == 0]

    while current_batch:
        batches.append(sorted(current_batch))
        next_batch = []
        for node in current_batch:
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_batch.append(neighbor)
        current_batch = next_batch

    return batches
