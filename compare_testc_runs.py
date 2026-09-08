from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _metrics_path(run: Path) -> Path:
    direct = run / "metrics_and_score.json"
    if direct.is_file():
        return direct
    latest = run / "testc_evaluation" / "latest.json"
    if latest.is_file():
        return Path(json.loads(latest.read_text(encoding="utf-8"))["metrics"])
    raise FileNotFoundError(f"Cannot locate Test_C metrics below {run}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two or more Test_C runs.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("testc_run_comparison.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.runs) < 2:
        raise ValueError("At least two Test_C runs are required")
    rows = []
    category_rows = []
    loaded: list[tuple[Path, dict, list[dict]]] = []
    for run in args.runs:
        path = _metrics_path(run.resolve())
        payload = json.loads(path.read_text(encoding="utf-8"))
        category_path = Path(payload["metrics_per_category"])
        categories_json = category_path.with_suffix(".json")
        categories = json.loads(categories_json.read_text(encoding="utf-8"))
        loaded.append((run, payload, categories))
    baseline_score = float(loaded[0][1]["score"]["total_score"])
    baseline_categories = {
        row["category"]: row for row in loaded[0][2]
    }
    for run, payload, categories in loaded:
        score = payload["score"]
        rows.append(
            {
                "run": str(run),
                "signature": payload.get("signature", ""),
                "total_score": score["total_score"],
                "delta_vs_first": float(score["total_score"]) - baseline_score,
                "s_cls": score["s_cls"],
                "s_seg": score["s_seg"],
                "s_zs": score["s_zs"],
                "seen_c_auroc": score["seen_macro"]["c_auroc"],
                "seen_c_ap": score["seen_macro"]["c_ap"],
                "seen_p_auroc": score["seen_macro"]["p_auroc"],
                "seen_p_ap": score["seen_macro"]["p_ap"],
                "seen_p_f1max": score["seen_macro"]["p_f1max"],
                "unseen_c_auroc": score["unseen_macro"]["c_auroc"],
                "unseen_c_ap": score["unseen_macro"]["c_ap"],
                "unseen_p_auroc": score["unseen_macro"]["p_auroc"],
                "unseen_p_ap": score["unseen_macro"]["p_ap"],
                "unseen_p_f1max": score["unseen_macro"]["p_f1max"],
                "category_eval_seconds": sum(float(row["seconds"]) for row in categories),
            }
        )
        for category in categories:
            base = baseline_categories[category["category"]]
            category_rows.append(
                {
                    "run": str(run),
                    "category": category["category"],
                    "partition": category["partition"],
                    "c_auroc": category["c_auroc"],
                    "delta_c_auroc": float(category["c_auroc"]) - float(base["c_auroc"]),
                    "c_ap": category["c_ap"],
                    "delta_c_ap": float(category["c_ap"]) - float(base["c_ap"]),
                    "p_auroc": category["p_auroc"],
                    "delta_p_auroc": float(category["p_auroc"]) - float(base["p_auroc"]),
                    "p_ap": category["p_ap"],
                    "delta_p_ap": float(category["p_ap"]) - float(base["p_ap"]),
                    "p_f1max": category["p_f1max"],
                    "delta_p_f1max": float(category["p_f1max"]) - float(base["p_f1max"]),
                    "seconds": category["seconds"],
                    "delta_seconds": float(category["seconds"]) - float(base["seconds"]),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    category_output = args.output.with_name(args.output.stem + "_per_category.csv")
    with category_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(category_rows[0]))
        writer.writeheader()
        writer.writerows(category_rows)
    markdown_output = args.output.with_suffix(".md")
    lines = [
        "# Test_C run comparison",
        "",
        "| Run | Total | Δ total | S_cls | S_seg | S_zs | Category eval seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {total_score:.4f} | {delta_vs_first:+.4f} | "
            "{s_cls:.4f} | {s_seg:.4f} | {s_zs:.4f} | "
            "{category_eval_seconds:.2f} |".format(**row)
        )
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(category_output.resolve())
    print(markdown_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
