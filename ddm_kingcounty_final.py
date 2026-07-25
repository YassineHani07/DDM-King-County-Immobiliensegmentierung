"""Finaler DDM-Code: King County Immobiliensegmentierung.

Team: Yassine Hani und Samuel Habte

Lege kc_house_data.csv oder die King-County-ZIP-Datei in denselben Ordner.
Das Skript vergleicht K-Means, Agglomerative Clustering und DBSCAN,
erstellt vier Business-Segmente und exportiert Tabellen und Grafiken.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "sqft_living", "log_sqft_lot", "bedrooms", "bathrooms", "floors",
    "house_age", "house_age_sq", "lat", "long", "grade", "condition",
    "view", "waterfront", "renovated",
]

COLORS = {
    "Neuere Familienhäuser": "#1F77B4",
    "Ältere Standardhäuser": "#2CA02C",
    "Waterfront-Luxus": "#F28E2B",
    "Renovierte Altbauten": "#9467BD",
}


def find_dataset(base_dir: Path) -> Path:
    """Findet kc_house_data.csv direkt oder extrahiert sie aus einer ZIP-Datei."""
    direct = base_dir / "kc_house_data.csv"
    if direct.exists():
        return direct

    matches = list(base_dir.rglob("kc_house_data.csv"))
    if matches:
        return matches[0]

    for zip_path in base_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as archive:
            members = [
                name for name in archive.namelist()
                if name.endswith("kc_house_data.csv") and not name.startswith("__MACOSX/")
            ]
            if members:
                target = base_dir / "data_extracted"
                target.mkdir(exist_ok=True)
                archive.extract(members[0], target)
                return target / members[0]

    raise FileNotFoundError(
        "kc_house_data.csv wurde nicht gefunden. Lade die CSV oder die ZIP-Datei hoch."
    )


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "id", "date", "price", "bedrooms", "bathrooms", "sqft_living",
        "sqft_lot", "floors", "waterfront", "view", "condition", "grade",
        "yr_built", "yr_renovated", "lat", "long",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten: {sorted(missing)}")

    data = df.copy()
    data.loc[data["bedrooms"] == 33, "bedrooms"] = 3

    # Bewusst konsistent mit der bisherigen Präsentation.
    data["house_age"] = 2015 - data["yr_built"]
    data["house_age_sq"] = data["house_age"] ** 2
    data["renovated"] = (data["yr_renovated"] > 0).astype(int)
    data["log_price"] = np.log1p(data["price"])  # nur Interpretation
    data["log_sqft_lot"] = np.log1p(data["sqft_lot"])
    return data


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    X = data[FEATURES].copy()
    if X.isna().any().any():
        raise ValueError("Clustering-Merkmale enthalten fehlende Werte.")
    scaled = StandardScaler().fit_transform(X)
    return pd.DataFrame(scaled, columns=FEATURES, index=data.index)


def sil(X, labels, sample_size: int = 5000) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan
    return silhouette_score(
        X, labels, sample_size=min(sample_size, len(labels)), random_state=RANDOM_STATE
    )


def evaluate_kmeans(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X)
        rows.append({
            "model": "K-Means", "parameter": f"k={k}", "n_clusters": k,
            "noise_share": 0.0, "silhouette": sil(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "inertia": model.inertia_,
        })
    return pd.DataFrame(rows)


def evaluate_agglomerative(X: pd.DataFrame, sample_size: int = 6000) -> pd.DataFrame:
    """Parametersuche auf Stichprobe, finales Modell später auf allen Daten."""
    X_eval = X.sample(n=min(sample_size, len(X)), random_state=RANDOM_STATE)
    rows = []
    for k in range(2, 11):
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X_eval)
        rows.append({
            "model": "Agglomerative", "parameter": f"k={k}, linkage=ward",
            "n_clusters": k, "noise_share": 0.0,
            "silhouette": sil(X_eval, labels),
            "davies_bouldin": davies_bouldin_score(X_eval, labels),
            "calinski_harabasz": calinski_harabasz_score(X_eval, labels),
            "evaluation_rows": len(X_eval),
        })
    return pd.DataFrame(rows)


def evaluate_dbscan(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for eps in [1.5, 2.0, 2.5, 3.0, 3.5]:
        for min_samples in [10, 25, 50]:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            mask = labels != -1
            valid_labels = labels[mask]
            valid_X = X.loc[mask]
            valid = n_clusters >= 2 and len(np.unique(valid_labels)) >= 2
            rows.append({
                "model": "DBSCAN",
                "parameter": f"eps={eps}, min_samples={min_samples}",
                "eps": eps, "min_samples": min_samples,
                "n_clusters": n_clusters,
                "noise_share": float(np.mean(labels == -1)),
                "silhouette": sil(valid_X, valid_labels) if valid else np.nan,
                "davies_bouldin": davies_bouldin_score(valid_X, valid_labels) if valid else np.nan,
                "calinski_harabasz": calinski_harabasz_score(valid_X, valid_labels) if valid else np.nan,
            })
    return pd.DataFrame(rows)


def fit_final_models(data: pd.DataFrame, X: pd.DataFrame) -> None:
    data["cluster_kmeans_3"] = KMeans(
        n_clusters=3, random_state=RANDOM_STATE, n_init=20
    ).fit_predict(X)

    # Auf allen 21.613 Zeilen; Google Colab wird empfohlen.
    data["cluster_agg_4"] = AgglomerativeClustering(
        n_clusters=4, linkage="ward"
    ).fit_predict(X)

    data["cluster_dbscan"] = DBSCAN(
        eps=3.0, min_samples=25
    ).fit_predict(X)


def compare_final_models(data: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    configs = [
        ("K-Means", "k=3", "cluster_kmeans_3", "Gute Basis; weniger differenzierte Segmente."),
        ("Agglomerative", "k=4, linkage=ward", "cluster_agg_4", "Am besten interpretierbar; vier klare Immobiliensegmente."),
        ("DBSCAN", "eps=3.0, min_samples=25", "cluster_dbscan", "Beste Silhouette- und Davies-Bouldin-Werte, aber ein sehr großer Hauptcluster."),
    ]
    rows = []
    for model, parameter, column, interpretation in configs:
        labels = data[column].to_numpy()
        mask = labels != -1
        valid_labels = labels[mask]
        valid_X = X.loc[mask]
        rows.append({
            "model": model, "parameter": parameter,
            "n_clusters": len(np.unique(valid_labels)),
            "noise_share": float(np.mean(labels == -1)),
            "silhouette": sil(valid_X, valid_labels),
            "davies_bouldin": davies_bouldin_score(valid_X, valid_labels),
            "calinski_harabasz": calinski_harabasz_score(valid_X, valid_labels),
            "business_interpretation": interpretation,
        })
    return pd.DataFrame(rows)


def derive_names(data: pd.DataFrame) -> dict[int, str]:
    """Cluster-IDs sind beliebig; Namen werden aus den Profilen abgeleitet."""
    p = data.groupby("cluster_agg_4").agg(
        count=("id", "count"),
        median_age=("house_age", "median"),
        renovated_rate=("renovated", "mean"),
        waterfront_rate=("waterfront", "mean"),
    )
    waterfront = int(p["waterfront_rate"].idxmax())
    renovated = int(p.drop(index=waterfront)["renovated_rate"].idxmax())
    remaining = [int(i) for i in p.index if i not in {waterfront, renovated}]
    newer = min(remaining, key=lambda i: p.loc[i, "median_age"])
    older = next(i for i in remaining if i != newer)
    return {
        newer: "Neuere Familienhäuser",
        older: "Ältere Standardhäuser",
        waterfront: "Waterfront-Luxus",
        renovated: "Renovierte Altbauten",
    }


def create_profile(data: pd.DataFrame) -> pd.DataFrame:
    data["cluster_name"] = data["cluster_agg_4"].map(derive_names(data))
    profile = data.groupby(["cluster_agg_4", "cluster_name"]).agg(
        count=("id", "count"), median_price=("price", "median"),
        median_sqft_living=("sqft_living", "median"),
        median_bedrooms=("bedrooms", "median"),
        median_bathrooms=("bathrooms", "median"),
        median_grade=("grade", "median"),
        median_condition=("condition", "median"),
        median_house_age=("house_age", "median"),
        renovated_rate=("renovated", "mean"),
        waterfront_rate=("waterfront", "mean"),
    ).round(2)
    profile["renovated_rate"] = (profile["renovated_rate"] * 100).round(1)
    profile["waterfront_rate"] = (profile["waterfront_rate"] * 100).round(1)
    return profile


def create_map(data: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 8))
    order = ["Neuere Familienhäuser", "Ältere Standardhäuser", "Renovierte Altbauten", "Waterfront-Luxus"]
    for name in order:
        part = data[data["cluster_name"] == name]
        special = name in {"Waterfront-Luxus", "Renovierte Altbauten"}
        ax.scatter(part["long"], part["lat"], s=24 if special else 8,
                   alpha=0.95 if special else 0.45, color=COLORS[name],
                   edgecolors="none", zorder=3 if special else 1)
    handles = [Line2D([0], [0], marker="o", color="white", label=name,
                      markerfacecolor=COLORS[name], markeredgecolor="none", markersize=9)
               for name in ["Neuere Familienhäuser", "Ältere Standardhäuser", "Waterfront-Luxus", "Renovierte Altbauten"]]
    ax.legend(handles=handles, title="Immobiliensegmente", loc="upper right",
              fontsize=10, title_fontsize=11, frameon=True)
    ax.set(xticks=[], yticks=[], xlabel="", ylabel="", title="")
    ax.grid(alpha=0.12)
    fig.tight_layout()
    path = OUTPUT_DIR / "agg_clustering_map_powerpoint.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def create_boxplot(data: pd.DataFrame) -> Path:
    order = ["Ältere Standardhäuser", "Neuere Familienhäuser", "Renovierte Altbauten", "Waterfront-Luxus"]
    labels = ["Ältere\nStandardhäuser", "Neuere\nFamilienhäuser", "Renovierte\nAltbauten", "Waterfront-\nLuxus"]
    values = [data.loc[data["cluster_name"] == name, "price"] / 1000 for name in order]
    fig, ax = plt.subplots(figsize=(11, 7))
    bp = ax.boxplot(values, tick_labels=labels, patch_artist=True, widths=0.58,
                    showfliers=False, medianprops={"color": "#222222", "linewidth": 2},
                    boxprops={"linewidth": 1.2, "edgecolor": "#555555"},
                    whiskerprops={"linewidth": 1.2, "color": "#555555"},
                    capprops={"linewidth": 1.2, "color": "#555555"})
    for box, name in zip(bp["boxes"], order):
        box.set_facecolor(COLORS[name]); box.set_alpha(0.68)
    medians = [data.loc[data["cluster_name"] == name, "price"].median() / 1000 for name in order]
    for pos, median in enumerate(medians, 1):
        label = (f"{median / 1000:.1f} Mio. $".replace(".", ",") if median >= 1000
                 else f"{median:,.0f} Tsd. $".replace(",", "."))
        ax.text(pos, median + 22, label, ha="center", va="bottom",
                fontsize=9, fontweight="semibold", color="#222222")
    ax.set_ylabel("Verkaufspreis in Tausend Dollar", fontsize=12)
    ax.set_xlabel(""); ax.set_title("")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}k".replace(",", ".")))
    ax.grid(axis="y", alpha=0.18, linewidth=0.8); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = OUTPUT_DIR / "agg_price_boxplot_powerpoint.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    csv_path = find_dataset(BASE_DIR)
    print("Datensatz:", csv_path)
    data = load_and_prepare(csv_path)
    print("Zeilen:", len(data), "| Fehlende Werte:", int(data.isna().sum().sum()))
    print("Preisspanne:", int(data.price.min()), "bis", int(data.price.max()), "US-Dollar")

    X = prepare_features(data)

    print("K-Means wird evaluiert ...")
    kmeans_results = evaluate_kmeans(X)
    print("Agglomerative Clustering wird evaluiert ...")
    agg_results = evaluate_agglomerative(X)
    print("DBSCAN wird evaluiert ...")
    dbscan_results = evaluate_dbscan(X)

    print("Finale Modelle werden trainiert ...")
    fit_final_models(data, X)
    comparison = compare_final_models(data, X)
    profile = create_profile(data)

    kmeans_results.to_csv(OUTPUT_DIR / "kmeans_evaluation.csv", index=False)
    agg_results.to_csv(OUTPUT_DIR / "agglomerative_evaluation.csv", index=False)
    dbscan_results.to_csv(OUTPUT_DIR / "dbscan_evaluation.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "final_model_comparison.csv", index=False)
    profile.to_csv(OUTPUT_DIR / "selected_clustering_profile.csv")
    data.to_csv(OUTPUT_DIR / "kingcounty_with_cluster_labels.csv", index=False)
    create_map(data)
    create_boxplot(data)

    print("\nFinaler Modellvergleich:")
    print(comparison.round(4).to_string(index=False))
    print("\nAusgewählte Clusterprofile:")
    print(profile.to_string())
    print("\nFertig. Ergebnisse:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
