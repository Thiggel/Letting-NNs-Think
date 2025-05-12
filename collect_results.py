import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    compute = np.array(sorted([float(k) for k in data.keys()])) * 100
    return data, compute


def extract_metric_series(data, compute, benchmark, metric_key='acc,none'):
    acc, stderr = [], []
    for comp_frac in sorted(data.keys(), key=lambda k: float(k)):
        entry = data[comp_frac].get(benchmark, {})
        acc.(entry.get(metric_key, np.nan))
        stderr.append(entry.get('acc_stderr,none', np.nan))
    return np.array(acc), np.array(stderr)


def compute_saved_at_retained(acc, compute, target_ratio=0.90):
    base_acc = acc[0]
    target_acc = base_acc * target_ratio
    return float(np.interp(target_acc, acc[::-1], compute[::-1]))


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment JSON')
    parser.add_argument('json_file', type=Path, help='Path to JSON file')
    args = parser.parse_args()

    data, compute = load_data(args.json_file)
    benchmarks = list(data[sorted(data.keys(), key=lambda k: float(k))[0]].keys())
    out_dir = args.json_file.with_suffix('').name
    Path(out_dir).mkdir(exist_ok=True)

    # 1) Compute saved @ 90%
    saved = {bm: compute_saved_at_retained(*extract_metric_series(data, compute, bm), 0.90)
             for bm in benchmarks}
    df_saved = pd.DataFrame.from_dict(saved, orient='index', columns=['Compute Saved (%)'])
    df_saved.to_csv(Path(out_dir)/'saved_at_90.csv')
    print(df_saved)

    # 2) Accuracy @0%
    acc0 = {bm: extract_metric_series(data, compute, bm)[0][0] for bm in benchmarks}
    df0 = pd.DataFrame.from_dict(acc0, orient='index', columns=['Accuracy @0%'])
    df0.to_csv(Path(out_dir)/'accuracy_at_0.csv')
    print(df0)

    # 3) Raw results with stderr
    idx = [f"{r:.0f}%" for r in compute]
    raw = pd.DataFrame(index=idx)
    for bm in benchmarks:
        acc, stderr = extract_metric_series(data, compute, bm)
        raw[bm] = [f"{a:.4f}\u00B1{s:.4f}" for a, s in zip(acc, stderr)]
    raw.to_csv(Path(out_dir)/'raw_results.csv')
    print(raw)

    # Print LaTeX table to stdout
    latex = raw.to_latex(escape=False)
    print("% LaTeX table for raw results")
    print(latex)

    # 4) Plots
    for bm in benchmarks:
        acc, _ = extract_metric_series(data, compute, bm)
        fig = go.Figure(go.Scatter(x=compute, y=acc, mode='lines+markers', line_shape='spline', name=bm))
        fig.update_layout(xaxis_title='Compute Omitted (%)', yaxis_title='Accuracy', template='simple_white')
        fig.write_image(str(Path(out_dir)/f"{bm}_curve.pdf"), engine='kaleido')
    print(f"Output in {out_dir}/")

if __name__=='__main__':
    main()

