"""Print the metrics saved by the training script."""
from pathlib import Path


if __name__ == "__main__":
    metrics_path = Path(__file__).resolve().parent.parent / "results" / "model_metrics.txt"
    print(metrics_path.read_text(encoding="utf-8"))
