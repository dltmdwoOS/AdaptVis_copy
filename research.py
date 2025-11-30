# research.py
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def check_is_correct(golden, gen: str) -> bool:
    if not isinstance(gen, str):
        return False
    cond1 = (golden in gen) or (golden.lower() in gen.lower())
    cond2 = not (golden.lower() == 'on' and 'front' in gen.strip().lower())
    return cond1 and cond2

def analyze_metric_for_pair(
    df,
    metric_col: str,
    low_weight: str,
    high_weight: str,
    valid_choices_col: str,
    strategy_type: str = "high_triggers_high",
):
    values = df[metric_col].values
    thresholds = np.linspace(values.min(), values.max(), 200)
    accuracies = []

    for th in thresholds:
        if strategy_type == "high_triggers_high":
            # metric >= T -> high_weight
            decisions = df[metric_col].apply(
                lambda x: high_weight if x >= th else low_weight
            )
        else:
            # metric <= T -> high_weight
            decisions = df[metric_col].apply(
                lambda x: high_weight if x <= th else low_weight
            )

        correct_count = sum(
            1 for dec, valid in zip(decisions, df[valid_choices_col]) if dec in valid
        )
        accuracies.append(correct_count / len(df))

    best_acc = max(accuracies)
    best_th = thresholds[int(np.argmax(accuracies))]
    return best_acc, best_th, thresholds, accuracies

def build_valid_choices_for_pair(df, low_weight: str, high_weight: str, colname: str):
    col_low = f"is_correct_{low_weight}"
    col_high = f"is_correct_{high_weight}"

    def _collect(row):
        choices = []
        if row[col_low]:
            choices.append(low_weight)
        if row[col_high]:
            choices.append(high_weight)
        return choices

    df[colname] = df.apply(_collect, axis=1)

def run_analysis_for_weight_pair(df, low_weight: str, high_weight: str, acc_baseline: float, save_prefix: str):
    print(f"\n===== Weight Pair: low={low_weight}, high={high_weight} =====")
    valid_col = f"valid_choices_{low_weight}_{high_weight}"

    # 1) valid_choices 생성
    build_valid_choices_for_pair(df, low_weight, high_weight, valid_col)

    # 2) Oracle
    acc_oracle = df[valid_col].apply(lambda xs: len(xs) > 0).mean()
    print(f"Oracle Accuracy for this pair: {acc_oracle:.4f}")

    # 3) Metric별 설정
    metric_configs = {
        "uncertainty_prob": "high_triggers_high",      # prob 낮을수록 불확실 -> high_weight
        "uncertainty_entropy": "low_triggers_high",  # entropy 높을수록 불확실 -> high_weight
        "uncertainty_jsd": "high_triggers_high",      # jsd 높을수록 불확실 -> high_weight
    }

    results = {}
    for metric, strategy in metric_configs.items():
        best_acc, best_th, ths, accs = analyze_metric_for_pair(
            df,
            metric_col=metric,
            low_weight=low_weight,
            high_weight=high_weight,
            valid_choices_col=valid_col,
            strategy_type=strategy,
        )
        results[metric] = {
            "best_acc": best_acc,
            "best_th": best_th,
            "thresholds": ths,
            "accuracies": accs,
        }
        print(
            f"[{metric}] Best Acc: {best_acc:.4f} | "
            f"Best Th: {best_th:.4f} | "
            f"Gain over Baseline(1.0): {(best_acc - acc_baseline) * 100:.2f}%p"
        )

    # 4) 그래프 저장
    plt.figure(figsize=(10, 7))
    plt.axhline(
        y=acc_baseline,
        color="gray",
        linestyle="--",
        label=f"Baseline 1.0 ({acc_baseline:.4f})",
        alpha=0.7,
    )
    plt.axhline(
        y=acc_oracle,
        color="black",
        linestyle=":",
        label=f"Oracle ({acc_oracle:.4f})",
        alpha=0.7,
    )

    colors = {
        "uncertainty_prob": "blue",
        "uncertainty_entropy": "red",
        "uncertainty_jsd": "purple",
    }

    for metric, res in results.items():
        plt.plot(
            res["thresholds"],
            res["accuracies"],
            label=f"{metric}",
            color=colors.get(metric, "green"),
        )
        plt.scatter(
            [res["best_th"]],
            [res["best_acc"]],
            color=colors.get(metric, "green"),
            zorder=5,
        )

    plt.title(f"Dynamic Selection (low={low_weight}, high={high_weight})")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = f"{save_prefix}_pair_{low_weight}_{high_weight}.png"
    plt.savefig(out_png)
    plt.close()
    print(f"Saved plot to {out_png}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-path",
        type=str,
        required=True,
        help="Path to JSON results from main_aro.py",
    )
    parser.add_argument(
        "--baseline-weight",
        type=str,
        default="1.0",
        help="Weight used as baseline (default: 1.0)",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="analysis",
        help="Prefix for saved plot filenames",
    )
    args = parser.parse_args()

    # 1. JSON 로드
    with open(args.results_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 2. is_correct_* 계산 (필요한 weight만)
    all_weights = ["0.5", "0.8", "1.0", "1.2", "1.5", "2.0"]
    for w in all_weights:
        col = f"is_correct_{w}"
        if col not in df.columns:
            df[col] = df.apply(
                lambda row: check_is_correct(row["Golden"], row["Generation"].get(w)),
                axis=1,
            )

    # 3. baseline
    base_col = f"is_correct_{args.baseline_weight}"
    acc_baseline = df[base_col].mean()
    print("=== Global Baseline ===")
    print(f"Baseline Weight: {args.baseline_weight}")
    print(f"Baseline Accuracy: {acc_baseline:.4f}")

    # 4. weight 조합 루프
    low_candidates = ["0.5", "0.8"]
    high_candidates = ["1.2", "1.5", "2.0"]

    all_results = {}
    for lw in low_candidates:
        for hw in high_candidates:
            res = run_analysis_for_weight_pair(
                df,
                low_weight=lw,
                high_weight=hw,
                acc_baseline=acc_baseline,
                save_prefix=args.save_prefix,
            )
            all_results[(lw, hw)] = res

    # 5. 요약 결과를 JSON으로 저장 (선택 사항)
    summary_out = args.save_prefix + "_summary.txt"
    with open(summary_out, "w") as f:
        f.write(f"Baseline weight: {args.baseline_weight}, acc={acc_baseline:.4f}\n")
        for (lw, hw), metrics in all_results.items():
            f.write(f"\nPair (low={lw}, high={hw}):\n")
            for mname, res in metrics.items():
                f.write(
                    f"  {mname}: best_acc={res['best_acc']:.4f}, "
                    f"best_th={res['best_th']:.6f}\n"
                )
    print(f"Saved summary to {summary_out}")

if __name__ == "__main__":
    main()
