import os
import json

import numpy as np
import matplotlib.pyplot as plt


def _load_results_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "results" not in data or not isinstance(data["results"], dict):
        raise ValueError(f"{path} does not contain a top-level 'results' dict.")
    return data["results"]


def aggregate_mean_acc_by_train_size(files_by_acq, train_sizes=None, acc_key="acc"):
    if train_sizes is None:
        train_sizes = list(range(20, 1020 + 1, 10))  # 101 points

    mean_acc_by_acq = {}

    for acq_name, paths in files_by_acq.items():
        acc_lists = {ts: [] for ts in train_sizes}

        for p in paths:
            results = _load_results_json(p)
            for ts in train_sizes:
                k = str(ts)
                if k in results and isinstance(results[k], dict) and acc_key in results[k]:
                    try:
                        v = float(results[k][acc_key])
                    except Exception:
                        continue
                    if np.isfinite(v):
                        acc_lists[ts].append(v)

        means = []
        for ts in train_sizes:
            vals = acc_lists[ts]
            means.append(float(np.mean(vals)) if len(vals) > 0 else None)

        mean_acc_by_acq[acq_name] = means

    return train_sizes, mean_acc_by_acq


def plot_mean_acc_curves_acquired(
    train_sizes,
    mean_acc_by_acq,
    out_path_png,
    y_limits_percent=(80.0, 100.0, 2),
    metric_name=None,
    title=None,
):
    out_dir = os.path.dirname(out_path_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    min_train = min(train_sizes)
    acquired = [ts - min_train for ts in train_sizes]  # 20->0, 1000->980

    plt.figure(figsize=(10, 6))
    x = np.array(acquired, dtype=np.float32)

    for acq_name, means in mean_acc_by_acq.items():
        y = np.array([np.nan if v is None else v for v in means], dtype=np.float32) * 100.0
        plt.plot(x, y, label=acq_name)

    plt.ylim(*(y_limits_percent[:2]))
    plt.yticks(np.arange(y_limits_percent[0], y_limits_percent[1], y_limits_percent[2]))

    plt.xlabel("# acquired images")
    plt.ylabel(metric_name)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=200)
    plt.close()


def aggregate_and_plot_only(files_by_acq, out_path_png, train_sizes=None, y_limits_percent=(80.0, 100.0, 2),
                            acc_key="acc", metric_name="Accuracy (%)", title="Mean Test Accuracy vs # Acquired Images"):
    train_sizes, mean_acc_by_acq = aggregate_mean_acc_by_train_size(
        files_by_acq=files_by_acq,
        train_sizes=train_sizes,
        acc_key=acc_key,
    )
    plot_mean_acc_curves_acquired(
        train_sizes=train_sizes,
        mean_acc_by_acq=mean_acc_by_acq,
        out_path_png=out_path_png,
        y_limits_percent=y_limits_percent,
        metric_name=metric_name,
        title=title,
    )


def main():
    files_by_acq = {
        "MatNormAnalytic": ["outs/matrixnormalanalyticmodel/PredictiveVarLogDet/seed_1025115162/results.json"],
        "MeanFieldVI": ["outs/meanfieldmodel/fixed_lik_var/PredictiveVarLogDet/seed_185306943/results.json"],
        "MatNormVI": ["outs/matrixnormalmodel/pretrain_then_vi/prior_eq_lik_cov=False/PredictiveVarLogDet/seed_629823440/results.json"],
    }
    """
    files_by_acq = {
        "EpistemicLogDet": ["outs/meanfieldmodel/not_fixed_lik_var/EpistemicLogDet/seed_1237511014/results.json"],
        "EpistemicMaxEigen": ["outs/meanfieldmodel/not_fixed_lik_var/EpistemicMaxEigen/seed_712439214/results.json"],
        "PredictiveLogDet": ["outs/meanfieldmodel/not_fixed_lik_var/PredictiveVarLogDet/seed_1156733818/results.json"],
        "PredictiveMaxEigen": ["outs/meanfieldmodel/not_fixed_lik_var/PredictiveVarMaxEigen/seed_1508111915/results.json"],
        "PredictiveTrace": ["outs/meanfieldmodel/not_fixed_lik_var/PredictiveVarTrace/seed_1689977356/results.json"],
        "Random": ["outs/meanfieldmodel/not_fixed_lik_var/Random/seed_931496582/results.json"],
    }
    """
    """
    files_by_acq = {
        "MaxEntropy": ["outs/bcnn/MaxPredictiveEntropy/seed_629408537/results.json",
                        "outs/bcnn/MaxPredictiveEntropy/seed_681707220/results.json",
                        "outs/bcnn/MaxPredictiveEntropy/seed_1747788007/results.json"],
        "Deterministic MaxEntropy": ["outs/dcnn/MaxPredictiveEntropy/seed_138900508/results.json",
                                    "outs/dcnn/MaxPredictiveEntropy/seed_285691879/results.json",
                                    "outs/dcnn/MaxPredictiveEntropy/seed_1330225670/results.json"]
    }
    """
    """
    files_by_acq = {
        "VarRatios": ["outs/bcnn/MaxVariationRatios/seed_864327772/results.json",
                      "outs/bcnn/MaxVariationRatios/seed_1342390706/results.json",
                      "outs/bcnn/MaxVariationRatios/seed_1345969693/results.json"],
        "Deterministic VarRatios": ["outs/dcnn/MaxVariationRatios/seed_945287710/results.json",
                                    "outs/dcnn/MaxVariationRatios/seed_1165722687/results.json",
                                    "outs/dcnn/MaxVariationRatios/seed_1851168182/results.json"]
    }
    """
    """
    files_by_acq = {
        "BALD": ["outs/bcnn/BALD/seed_123694267/results.json",
                 "outs/bcnn/BALD/seed_802820914/results.json",
                 "outs/bcnn/BALD/seed_841345976/results.json"],
        "Deterministic BALD (Random)": ["outs/dcnn/BALD/seed_726859865/results.json",
                                        "outs/dcnn/BALD/seed_1788123464/results.json",
                                        "outs/dcnn/BALD/seed_2017416269/results.json"]
    }
    """
    """
    files_by_acq = {
        "BALD": ["outs/bcnn/BALD/seed_123694267/results.json",
                 "outs/bcnn/BALD/seed_802820914/results.json",
                 "outs/bcnn/BALD/seed_841345976/results.json"],
        "MaxEntropy": ["outs/bcnn/MaxPredictiveEntropy/seed_629408537/results.json",
                                 "outs/bcnn/MaxPredictiveEntropy/seed_681707220/results.json",
                                 "outs/bcnn/MaxPredictiveEntropy/seed_1747788007/results.json"],
        "MeanSTD": ["outs/bcnn/MaxMeanStandardDeviation/seed_9867169/results.json",
                    "outs/bcnn/MaxMeanStandardDeviation/seed_787355126/results.json",
                    "outs/bcnn/MaxMeanStandardDeviation/seed_1137885420/results.json"],
        "VarRatios": ["outs/bcnn/MaxVariationRatios/seed_864327772/results.json",
                      "outs/bcnn/MaxVariationRatios/seed_1342390706/results.json",
                      "outs/bcnn/MaxVariationRatios/seed_1345969693/results.json"],
        "Random": ["outs/bcnn/Random/seed_156735205/results.json",
                   "outs/bcnn/Random/seed_1649166732/results.json",
                   "outs/bcnn/Random/seed_1944605677/results.json"],
    }
    """

    out_path_png = "figs/new_baseline_comparison.png"

    train_sizes = list(range(20, 1020 + 1, 10))
    y_limits_percent = (6.0, 30.0, 2)

    aggregate_and_plot_only(
        files_by_acq=files_by_acq,
        out_path_png=out_path_png,
        train_sizes=train_sizes,
        y_limits_percent=y_limits_percent,
        acc_key="rmse",
        metric_name="RMSE",
        title="Test RMSE vs # Acquired Images"
    )


if __name__ == "__main__":
    main()
