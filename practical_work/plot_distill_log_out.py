import re
import ast
import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/test_distill.sbatch-44106792.out"


def try_parse_dict(line):
    """
    Safely parse dictionary-like log lines.
    """
    try:
        return ast.literal_eval(line)
    except:
        return None


def load_logs(path):
    rows = []

    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()

            # skip headers / junk
            if not line.startswith("{") or "train_runtime" in line:
                continue

            d = try_parse_dict(line)
            if d is None:
                continue

            # flatten useful fields
            row = {}

            # numeric fields
            for k, v in d.items():
                try:
                    row[k] = float(v)
                except:
                    row[k] = v

            row["step"] = i
            rows.append(row)

    return pd.DataFrame(rows)


def smooth(series, window=200):
    return series.rolling(window, min_periods=1).mean()


def plot(df):
    plt.figure(figsize=(16, 10))

    # ---------------------------
    # 1. TOTAL LOSS
    # ---------------------------
    if "loss" in df:
        plt.subplot(2, 2, 1)
        plt.plot(df["step"], df["loss"], alpha=0.3, label="raw")
        plt.plot(df["step"], smooth(df["loss"]), label="smoothed")
        plt.title("Total Loss")
        plt.legend()

    # ---------------------------
    # 2. DISTILLATION COMPONENTS
    # ---------------------------
    plt.subplot(2, 2, 2)

    for key in [
        "distillation_loss/1_cross_entropy",
        "distillation_loss/2_kl",
        "distillation_loss/3_hs_cosine"
    ]:
        if key in df:
            plt.plot(df["step"], smooth(df[key]), label=key)

    plt.title("Distillation Loss Components (smoothed)")
    plt.legend()

    # ---------------------------
    # 3. LEARNING RATE
    # ---------------------------
    if "learning_rate" in df:
        plt.subplot(2, 2, 3)
        plt.plot(df["step"], df["learning_rate"])
        plt.title("Learning Rate Schedule")
        plt.yscale("log")

    # ---------------------------
    # 4. GRAD NORM
    # ---------------------------
    if "grad_norm" in df:
        plt.subplot(2, 2, 4)
        plt.plot(df["step"], smooth(df["grad_norm"], 200))
        plt.title("Gradient Norm (smoothed)")
        plt.yscale("log")

    plt.tight_layout()
    plt.show()


def main():
    print("Loading logs...")
    df = load_logs(LOG_PATH)

    print(f"Parsed rows: {len(df)}")

    # sanity checks
    print(df.columns)

    plot(df)


if __name__ == "__main__":
    main()