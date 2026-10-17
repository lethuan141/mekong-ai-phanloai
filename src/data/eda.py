
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[2]
    data_path = root / "data" / "mau" / "issue_descriptions.csv"
    output_dir = root / "docs"
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    required_columns = {"issue_desc", "issue_category"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset thiếu cột: {missing}")

    # Thống kê dữ liệu
    print("=== TONG QUAN DU LIEU ===")
    print(f"So dong, so cot: {df.shape}")
    print(f"So dong trung lap: {df.duplicated().sum()}")
    print("\n=== PHAN BO NHAN ===")

    counts = df["issue_category"].value_counts()
    percentages = (
        df["issue_category"].value_counts(normalize=True) * 100
    ).round(2)

    summary = pd.DataFrame({
        "So mau": counts,
        "Ty le (%)": percentages,
    })
    print(summary.to_string())

    print("\n=== DO DAI MO TA ===")
    lengths = df["issue_desc"].fillna("").astype(str).str.split().str.len()
    print(lengths.describe().round(2).to_string())
    print(f"Ty le mo ta duoi 10 tu: {(lengths < 10).mean() * 100:.2f}%")

    # Biểu đồ phân bố nhãn
    plt.figure(figsize=(9, 5))
    counts.sort_values().plot(kind="barh")
    plt.title("Phan bo nhan su co")
    plt.xlabel("So luong mau")
    plt.ylabel("Nhan")
    plt.tight_layout()

    chart_path = output_dir / "eda-phan-bo-nhan.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nDa luu bieu do: {chart_path}")
    print("EDA hoan tat.")


if __name__ == "__main__":
    main()
