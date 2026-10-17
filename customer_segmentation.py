# Customer segmentation analysis for the Mall Customers dataset.

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = Path(r"C:\Users\Himanshi Verma\Downloads\Mall_Customers.csv")
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input Mall_Customers.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "customer_segmentation_output")
    return parser.parse_args()


def profile_name(income: float, spending: float, income_cutoffs: np.ndarray, spend_cutoffs: np.ndarray) -> str:
    def level(value: float, cutoffs: np.ndarray) -> str:
        return "Low" if value < cutoffs[0] else "Medium" if value < cutoffs[1] else "High"

    income_level = f"{level(income, income_cutoffs)} Income"
    spend_level = f"{level(spending, spend_cutoffs)} Spend"
    return f"{income_level} / {spend_level}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    required = {"CustomerID", "Gender", "Age", *FEATURES}
    missing_cols = required.difference(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_cols))}")

    # Use income and spending for segmentation; retain age and gender for interpretation.
    raw_features = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    preprocess = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    scaled = preprocess.fit_transform(raw_features)

    metric_rows: list[dict[str, float | int | str]] = []
    for k in range(2, 11):
        for name, model in (
            ("KMeans", KMeans(n_clusters=k, n_init=20, random_state=42)),
            ("Agglomerative", AgglomerativeClustering(n_clusters=k, linkage="ward")),
        ):
            labels = model.fit_predict(scaled)
            row: dict[str, float | int | str] = {
                "Algorithm": name,
                "Clusters": k,
                "Silhouette": float(silhouette_score(scaled, labels)),
            }
            if name == "KMeans":
                row["Inertia"] = float(model.inertia_)
            metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    km_metrics = metrics[metrics["Algorithm"] == "KMeans"].copy()
    selected_k = int(km_metrics.loc[km_metrics["Silhouette"].idxmax(), "Clusters"])
    chosen_score = float(km_metrics.loc[km_metrics["Clusters"] == selected_k, "Silhouette"].iloc[0])

    final_model = KMeans(n_clusters=selected_k, n_init=20, random_state=42)
    cluster_ids = final_model.fit_predict(scaled)
    segmented = data.copy()
    segmented["Cluster"] = cluster_ids

    profiles = segmented.groupby("Cluster").agg(
        Customers=("CustomerID", "count"),
        Age_Mean=("Age", "mean"),
        Income_Mean=(FEATURES[0], "mean"),
        Spending_Mean=(FEATURES[1], "mean"),
    )
    # Rank cluster means into low/middle/high bands to distinguish the center cluster.
    income_cutoffs = np.quantile(profiles["Income_Mean"], [1 / 3, 2 / 3])
    spend_cutoffs = np.quantile(profiles["Spending_Mean"], [1 / 3, 2 / 3])
    profiles["Segment"] = [
        profile_name(row.Income_Mean, row.Spending_Mean, income_cutoffs, spend_cutoffs)
        for row in profiles.itertuples()
    ]
    name_map = profiles["Segment"].to_dict()
    segmented["Segment"] = segmented["Cluster"].map(name_map)
    profiles["Share_Pct"] = profiles["Customers"] / len(segmented) * 100
    profiles["Female_Pct"] = segmented.groupby("Cluster")["Gender"].apply(
        lambda s: (s.astype(str).str.casefold() == "female").mean() * 100
    )
    profiles["Age_Mean"] = profiles["Age_Mean"].round(1)
    profiles[["Income_Mean", "Spending_Mean", "Share_Pct", "Female_Pct"]] = profiles[
        ["Income_Mean", "Spending_Mean", "Share_Pct", "Female_Pct"]
    ].round(1)

    metrics.to_csv(args.output_dir / "cluster_metrics.csv", index=False)
    profiles.reset_index().to_csv(args.output_dir / "segment_profiles.csv", index=False)
    segmented.to_csv(args.output_dir / "customers_with_segments.csv", index=False)

    make_selection_plot(metrics, selected_k, args.output_dir / "cluster_selection.png")
    make_segment_plot(segmented, profiles, args.output_dir / "customer_segments.png")
    write_report(args.input, args.output_dir, len(data), raw_features.isna().sum(), metrics, profiles, selected_k, chosen_score)
    print(f"Analysed {len(data)} customers; selected KMeans with k={selected_k} (silhouette={chosen_score:.3f}).")
    print(f"Outputs: {args.output_dir.resolve()}")


def make_selection_plot(metrics: pd.DataFrame, selected_k: int, output: Path) -> None:
    km = metrics[metrics["Algorithm"] == "KMeans"]
    ag = metrics[metrics["Algorithm"] == "Agglomerative"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(km["Clusters"], km["Inertia"], marker="o", color="#2563eb")
    axes[0].axvline(selected_k, color="#dc2626", linestyle="--", label=f"Selected k={selected_k}")
    axes[0].set(title="KMeans elbow curve", xlabel="Number of clusters (k)", ylabel="Inertia")
    axes[0].legend(frameon=False)
    axes[1].plot(km["Clusters"], km["Silhouette"], marker="o", label="KMeans", color="#2563eb")
    axes[1].plot(ag["Clusters"], ag["Silhouette"], marker="s", label="Agglomerative", color="#16a34a")
    axes[1].axvline(selected_k, color="#dc2626", linestyle="--")
    axes[1].set(title="Silhouette comparison", xlabel="Number of clusters (k)", ylabel="Silhouette score", xticks=km["Clusters"])
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_segment_plot(data: pd.DataFrame, profiles: pd.DataFrame, output: Path) -> None:
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster, group in data.groupby("Cluster"):
        ax.scatter(group[FEATURES[0]], group[FEATURES[1]], s=45, alpha=0.72,
                   color=colors(int(cluster)), label=profiles.loc[cluster, "Segment"])
    centers = profiles[["Income_Mean", "Spending_Mean"]]
    ax.scatter(centers["Income_Mean"], centers["Spending_Mean"], marker="X", s=220,
               c="black", edgecolor="white", linewidth=1.1, label="Segment average")
    ax.set(title="Customer segments by income and spending", xlabel="Annual income (k$)", ylabel="Spending score (1–100)")
    ax.legend(title="Segment", frameon=True, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(input_path: Path, output_dir: Path, n: int, missing: pd.Series,
                 metrics: pd.DataFrame, profiles: pd.DataFrame, selected_k: int, score: float) -> None:
    best_ag = metrics[metrics["Algorithm"] == "Agglomerative"].sort_values("Silhouette", ascending=False).iloc[0]
    lines = [
        "# Customer Segmentation Analysis",
        "",
        f"**Dataset:** `{input_path}`  ",
        f"**Customers analysed:** {n}  ",
        f"**Selected method:** KMeans, k={selected_k} (silhouette {score:.3f})  ",
        "**Clustering features:** annual income and spending score, median-imputed and standardized.",
        "",
        "## Method and cluster selection",
        "",
        "CustomerID was excluded because it is an identifier. Age and gender were held out of clustering and used to describe the resulting groups. KMeans and Ward-linkage agglomerative clustering were compared for k=2 to 10 using silhouette score; KMeans inertia was also reviewed with the elbow curve. The selected k is the KMeans option with the highest silhouette in that range.",
        "",
        f"The best agglomerative result in the same range was k={int(best_ag['Clusters'])} with silhouette {float(best_ag['Silhouette']):.3f}. Silhouette is a separation diagnostic, not proof that customer behavior has discrete natural boundaries.",
        "",
        "![Elbow and silhouette comparison](cluster_selection.png)",
        "",
        "## Segment profiles and business actions",
        "",
        "| Segment | Customers | Share | Avg. age | Avg. income (k$) | Avg. spend score | Female share | Suggested action |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    actions = {
        "High Income / High Spend": "Reward loyalty; test premium and early-access offers.",
        "High Income / Low Spend": "Test relevant discovery and premium-category recommendations.",
        "Medium Income / Medium Spend": "Use personalized cross-sells and measure response before scaling.",
        "Low Income / High Spend": "Encourage repeat visits with value bundles; monitor offer economics.",
        "Low Income / Low Spend": "Use low-cost re-engagement and entry-price offers; avoid expensive incentives.",
    }
    for row in profiles.itertuples():
        segment = row.Segment
        action = actions.get(segment, "Test a targeted offer and measure incremental spend.")
        lines.append(
            f"| {segment} | {row.Customers} | {row.Share_Pct:.1f}% | {row.Age_Mean:.1f} | {row.Income_Mean:.1f} | {row.Spending_Mean:.1f} | {row.Female_Pct:.1f}% | {action} |"
        )
    lines += [
        "",
        "![Customer segment scatter plot](customer_segments.png)",
        "",
        "## Data quality and caveats",
        "",
        f"Missing values in clustering inputs before imputation: annual income {int(missing[FEATURES[0]])}; spending score {int(missing[FEATURES[1]])}. The preprocessing pipeline uses median imputation followed by standard scaling. Segment names describe low, middle, and high bands of cluster-average income and spending across the selected clusters.",
        "",
        "Spending Score is a dataset-provided score, not observed transaction history, so these results support targeting hypotheses rather than causal claims. Validate offers with campaign tests and refresh segments as purchasing data changes. Demographic summaries are descriptive and should not be used as a substitute for testing customer response.",
        "",
        "## Files",
        "",
        "- `customers_with_segments.csv`: original customer records plus cluster and readable segment labels.",
        "- `segment_profiles.csv`: cluster size and average demographic/spending profile.",
        "- `cluster_metrics.csv`: KMeans and agglomerative silhouette scores; KMeans inertia.",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
