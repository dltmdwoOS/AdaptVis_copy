# research.py
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 정답 판별 함수 (기존과 동일)
def check_is_correct(golden, gen: str) -> bool:
    if not isinstance(gen, str):
        return False
    cond1 = (golden in gen) or (golden.lower() in gen.lower())
    cond2 = not (golden.lower() == 'on' and 'front' in gen.strip().lower())
    return cond1 and cond2

# 2. 분석 로직 (Threshold Sweep)
def analyze_metric_for_pair(
    df,
    metric_col: str,
    low_weight: str,
    high_weight: str,
    valid_choices_col: str,
    strategy_type: str = "high_triggers_high",
):
    """
    metric_col: 'uncertainty_prob' (confidence), 'uncertainty_entropy', 'uncertainty_jsd' 등
    strategy_type:
      - 'high_triggers_high': 값 >= T 이면 high_weight 선택 (Entropy, JSD 등)
      - 'low_triggers_high' : 값 <= T 이면 high_weight 선택 (Confidence 등)
    """
    values = df[metric_col].values
    # 값이 NaN인 경우 방어 로직 (혹시 모를 데이터 누락 대비)
    if np.isnan(values).all():
        return 0.0, 0.0, [], []
        
    thresholds = np.linspace(values.min(), values.max(), 200)
    accuracies = []

    for th in thresholds:
        if strategy_type == "high_triggers_high":
            # 불확실성(metric)이 높으면 -> High Temp (Explore)
            decisions = df[metric_col].apply(
                lambda x: high_weight if x >= th else low_weight
            )
        else:
            # 확신(Confidence)이 낮으면 -> High Temp (Explore)
            decisions = df[metric_col].apply(
                lambda x: high_weight if x <= th else low_weight
            )

        # 결정된 weight가 valid_choices(정답 후보군)에 포함되는지 확인
        correct_count = sum(
            1 for dec, valid in zip(decisions, df[valid_choices_col]) if dec in valid
        )
        accuracies.append(correct_count / len(df))

    best_acc = max(accuracies)
    best_th = thresholds[int(np.argmax(accuracies))]
    return best_acc, best_th, thresholds, accuracies

# 3. Valid Choices 생성기
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

# 4. 실행 및 시각화 함수
def run_analysis_for_weight_pair(df, low_weight: str, high_weight: str, acc_baseline: float, save_prefix: str):
    print(f"\n===== Weight Pair: low={low_weight}, high={high_weight} =====")
    valid_col = f"valid_choices_{low_weight}_{high_weight}"

    # valid_choices 생성
    build_valid_choices_for_pair(df, low_weight, high_weight, valid_col)

    # Oracle 계산
    acc_oracle = df[valid_col].apply(lambda xs: len(xs) > 0).mean()
    print(f"Oracle Accuracy for this pair: {acc_oracle:.4f}")

    # 지표별 전략 설정
    # 데이터 키 이름에 주의: JSON 키는 'confidence', 'entropy', 'jsd' 등임
    # 여기서는 DataFrame 컬럼명을 'uncertainty_prob' 등으로 매핑해서 쓸 예정
    metric_configs = {
        "uncertainty_prob": "high_triggers_high",      # confidence 낮으면 high temp
        "uncertainty_entropy": "low_triggers_high",  # entropy 높으면 high temp
        "uncertainty_jsd": "high_triggers_high",      # jsd 높으면 high temp
    }

    results = {}
    for metric, strategy in metric_configs.items():
        # 해당 컬럼이 존재하는지 확인
        if metric not in df.columns:
            print(f"Skipping {metric} (Not found in DataFrame)")
            continue
            
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

    # 그래프 그리기
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
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument("--baseline-weight", type=str, default="1.0")
    parser.add_argument("--save-prefix", type=str, default="analysis")
    args = parser.parse_args()

    # 1. Load JSON
    with open(args.results_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # === [수정됨] 2. Uncertainties 컬럼 추출 (Flattening) ===
    # JSON 구조: "Uncertainties": {"confidence": ..., "entropy": ..., "jsd": ...}
    def extract_uncertainties(row):
        unc = row.get('Uncertainties', {})
        if not isinstance(unc, dict):
            return pd.Series([0, 0, 0], index=['uncertainty_prob', 'uncertainty_entropy', 'uncertainty_jsd'])
        
        return pd.Series({
            'uncertainty_prob': unc.get('confidence', 0),
            'uncertainty_entropy': unc.get('entropy', 0),
            'uncertainty_jsd': unc.get('jsd', 0)
        })

    df[['uncertainty_prob', 'uncertainty_entropy', 'uncertainty_jsd']] = df.apply(extract_uncertainties, axis=1)

    # === [수정됨] 3. is_correct_* 계산 (Nested Dict Access) ===
    # JSON 구조: "Generation_map": {"1.0": {"generation": "...", ...}, ...}
    all_weights = ["0.5", "0.8", "1.0", "1.2", "1.5", "2.0"]
    for w in all_weights:
        col = f"is_correct_{w}"
        if col not in df.columns:
            df[col] = df.apply(
                lambda row: check_is_correct(
                    row["Golden"], 
                    row.get("Generation_map", {}).get(w, {}).get("generation", "")
                ),
                axis=1,
            )

    # 4. Baseline Calculation
    base_col = f"is_correct_{args.baseline_weight}"
    if base_col not in df.columns:
        print(f"Warning: Baseline weight {args.baseline_weight} not found. Using 1.0 if available.")
        args.baseline_weight = "1.0"
        base_col = "is_correct_1.0"

    acc_baseline = df[base_col].mean()
    print("=== Global Baseline ===")
    print(f"Baseline Weight: {args.baseline_weight}")
    print(f"Baseline Accuracy: {acc_baseline:.4f}")

    # 5. Weight Pair Loop
    low_candidates = ["0.5", "0.8"]
    high_candidates = ["1.2", "1.5", "2.0"]

    all_results = {}
    for lw in low_candidates:
        for hw in high_candidates:
            # 해당 weight 데이터가 실제로 존재하는지 체크
            if f"is_correct_{lw}" not in df.columns or f"is_correct_{hw}" not in df.columns:
                print(f"Skipping pair ({lw}, {hw}) due to missing data columns.")
                continue
                
            res = run_analysis_for_weight_pair(
                df,
                low_weight=lw,
                high_weight=hw,
                acc_baseline=acc_baseline,
                save_prefix=args.save_prefix,
            )
            all_results[(lw, hw)] = res

    # 6. Summary Save
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
