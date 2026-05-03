from __future__ import annotations

import argparse

import torch

from sensenova_u1.utils import ModelParamInspector, build_rules, format_bytes, format_param_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count actual model parameters and split by functional groups. "
            "Default groups are tuned for SenseNova-U1-8B-MoT."
        )
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Local checkpoint path or HuggingFace model id.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float32", "float16", "bfloat16"),
        help=(
            "Load dtype. bfloat16 by default to align with inference scripts. "
            "Note: dtype only affects load-time memory; param counts are dtype-independent."
        ),
    )
    parser.add_argument(
        "--custom_groups_json",
        default=None,
        help=('Optional JSON file to override grouping rules. Format: {"group_name": ["prefix1", "prefix2"]}.'),
    )
    parser.add_argument(
        "--show_groups",
        default="shared",
        help=(
            "Comma-separated group names whose member parameters will be listed in detail. "
            "Use 'all' for every group, or empty string to disable. Default: shared."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    inspector = ModelParamInspector(
        model_path=args.model_path,
        dtype=dtype_map[args.dtype],
    )
    rules = build_rules(args.custom_groups_json)
    result = inspector.count(rules)

    name_w = 28
    width = name_w + 1 + 12 + 1 + 16 + 1 + 10  # = 69
    dtype_label = args.dtype
    memory_header = f"memory ({dtype_label})"
    print(f"Model: {result.model_path}")
    print(f"Load dtype:   {dtype_label}")
    print(f"Total params: {format_param_count(result.total_params)}")
    print(f"Total memory: {format_bytes(result.total_bytes)} ({dtype_label})")
    print("-" * width)
    print(f"{'group':<{name_w}} {'params':>12} {memory_header:>16} {'ratio':>10}")
    print("-" * width)
    for group in result.groups:
        ratio = (group.params / result.total_params) * 100.0 if result.total_params else 0.0
        print(
            f"{group.name:<{name_w}} {format_param_count(group.params):>12} "
            f"{format_bytes(group.bytes):>16} {ratio:>9.2f}%"
        )

    _print_pathway_summary(result, width=width, name_w=name_w, memory_header=memory_header)
    _print_group_entries(result, args.show_groups, width=width, dtype_label=dtype_label)


def _print_pathway_summary(result, *, width: int, name_w: int, memory_header: str) -> None:
    """Forward-pathway coverage: parameters touched when running each task.

    Each pathway sums its dedicated transformer with the shared text I/O.
    Both embed_tokens and lm_head are exercised by both tasks: the latter is
    used by t2i-reasoning during the thinking phase before image tokens are
    emitted, so it is a real shared component, not understanding-only.
    """
    by_name = {g.name: g for g in result.groups}
    required = ("understanding_transformer", "generation_transformer", "shared")
    if not all(k in by_name for k in required):
        return

    shared = by_name["shared"]
    pathways = (
        (
            "understanding pathway",
            by_name["understanding_transformer"].params + shared.params,
            by_name["understanding_transformer"].bytes + shared.bytes,
        ),
        (
            "generation pathway",
            by_name["generation_transformer"].params + shared.params,
            by_name["generation_transformer"].bytes + shared.bytes,
        ),
    )

    print("-" * width)
    print("Pathway breakdown (shared counted in both):")
    print("-" * width)
    print(f"{'pathway':<{name_w}} {'params':>12} {memory_header:>16} {'ratio':>10}")
    print("-" * width)
    for name, params, nbytes in pathways:
        ratio = (params / result.total_params) * 100.0 if result.total_params else 0.0
        print(f"{name:<{name_w}} {format_param_count(params):>12} {format_bytes(nbytes):>16} {ratio:>9.2f}%")


def _print_group_entries(result, show_groups_arg: str, *, width: int, dtype_label: str) -> None:
    """Dump member parameters for the requested groups."""
    show_groups_arg = (show_groups_arg or "").strip()
    if not show_groups_arg:
        return

    by_name = {g.name: g for g in result.groups}
    if show_groups_arg.lower() == "all":
        target_names = [g.name for g in result.groups]
    else:
        target_names = [n.strip() for n in show_groups_arg.split(",") if n.strip()]

    for group_name in target_names:
        group = by_name.get(group_name)
        if group is None:
            print()
            print(f"[show_groups] group '{group_name}' not found, skipped.")
            continue

        print()
        print("-" * width)
        print(
            f"Members of group '{group.name}' "
            f"({len(group.entries)} params, {format_param_count(group.params)} total, "
            f"{format_bytes(group.bytes)} @ {dtype_label})"
        )
        print("-" * width)
        print(f"{'param name':<54} {'numel':>10} {'dtype':>8}")
        print("-" * width)
        for entry in group.entries:
            print(f"{entry.name:<54} {format_param_count(entry.numel):>10} {entry.dtype:>8}")


if __name__ == "__main__":
    main()
