import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_data(json_path: Path):
    """Load JSON experiment data, extract actual compute-saved percentages, and clean raw data."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Sort keys by numeric value
    comp_keys = sorted(data.keys(), key=lambda k: float(k))
    compute = []
    for k in comp_keys:
        # Use measured percent_tokens_skipped if available, else fallback to key
        pct = data[k].pop("percent_tokens_skipped", None)
        if pct is None:
            pct = float(k)
        compute.append(pct * 100)
    return data, np.array(compute)


def extract_metric_series(
    data: dict, compute: np.ndarray, benchmark: str, metric_key: str = "acc,none"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract accuracy and stderr arrays for a given benchmark across thresholds.
    """
    acc_list, stderr_list = [], []
    for comp_frac in sorted(data.keys(), key=lambda k: float(k)):
        entry = data[comp_frac].get(benchmark, {})
        acc_list.append(entry.get(metric_key, np.nan))
        stderr_list.append(entry.get("acc_stderr,none", np.nan))
    return np.array(acc_list), np.array(stderr_list)


def compute_saved_at_retained(
    acc: np.ndarray, compute: np.ndarray, target_ratio: float = 0.90
) -> float:
    """
    Compute the compute-saved percentage at which accuracy falls to target_ratio of the base.
    """
    base_acc = acc[0]
    target_acc = base_acc * target_ratio
    return float(np.interp(target_acc, acc[::-1], compute[::-1]))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment JSON and summarize results"
    )
    parser.add_argument(
        "json_file", type=Path, help="Path to JSON file with experiment data"
    )
    args = parser.parse_args()

    data, compute = load_data(args.json_file)
    # Identify benchmarks by filtering out non-dict entries and unwanted mmlu suites
    first_key = sorted(data.keys(), key=lambda k: float(k))[0]
    benchmarks = []
    for bm, val in data[first_key].items():
        if not isinstance(val, dict):
            continue
        if bm.startswith("mmlu") and bm != "mmlu_stem":
            continue
        benchmarks.append(bm)

    out_dir = args.json_file.with_suffix("").name
    Path(out_dir).mkdir(exist_ok=True)

    # 1) Compute Saved @90%
    saved = {}
    for bm in benchmarks:
        acc, _ = extract_metric_series(data, compute, bm)
        saved[bm] = compute_saved_at_retained(acc, compute, target_ratio=0.90)
    df_saved = pd.DataFrame.from_dict(
        saved, orient="index", columns=["Compute Saved @90%"]
    )
    df_saved.index.name = "Benchmark"
    df_saved.to_csv(Path(out_dir) / "saved_at_90.csv")
    print("=== Compute Saved @90% ===")
    print(df_saved)

    # 2) Accuracy @0% compute omitted (as percent)
    acc0 = {}
    for bm in benchmarks:
        acc, _ = extract_metric_series(data, compute, bm)
        acc0[bm] = acc[0] * 100
    df_acc0 = pd.DataFrame.from_dict(acc0, orient="index", columns=["Accuracy @0%"])
    df_acc0.index.name = "Benchmark"
    df_acc0.to_csv(Path(out_dir) / "accuracy_at_0.csv")
    print("\n=== Accuracy @0% Compute Omitted ===")
    print(df_acc0)

    # 3) Raw results table with acc±stderr (as percent)
    idx_labels = [f"{int(c)}%" for c in compute]
    raw = pd.DataFrame(index=idx_labels)
    raw.index.name = "Compute Saved (%)"
    for bm in benchmarks:
        acc, stderr = extract_metric_series(data, compute, bm)
        raw[bm] = [f"{a*100:.2f}\u00B1{s*100:.2f}" for a, s in zip(acc, stderr)]
    raw.to_csv(Path(out_dir) / "raw_results.csv")
    print("\n=== Raw Results ===")
    print(raw)

    # Print LaTeX
    summary = df_saved.join(df_acc0)
    print("\n% LaTeX: Summary table")
    # format Compute Saved @90% as two decimals, Accuracy @0% already percent
    print(summary.to_latex(float_format="%.2f"))

    print("\n% LaTeX: Raw results table")
    print(raw.to_latex(escape=False))

    # 4) Plots: Accuracy vs Compute Saved
    for bm in benchmarks:
        acc, _ = extract_metric_series(data, compute, bm)
        fig = go.Figure(
            go.Scatter(
                x=compute,
                y=acc * 100,
                mode="lines+markers",
                line_shape="spline",
                name=bm,
            )
        )
        fig.update_layout(
            xaxis_title="Compute Saved (%)",
            yaxis_title="Accuracy (%)",
            template="simple_white",
        )
        fig.update_yaxes(rangemode="tozero")

        out_path = Path(out_dir) / f"{bm}_accuracy_vs_saved.pdf"
        fig.write_image(str(out_path), engine="kaleido")
    print(f"Plots written to '{out_dir}/' directory.")


if __name__ == "__main__":
    main()
