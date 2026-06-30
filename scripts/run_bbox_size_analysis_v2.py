from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2_bbox_size_analysis import run_bbox_size_analysis


def main() -> None:
    analysis = run_bbox_size_analysis("trio_front")
    print("BBox-size metrics:")
    print(analysis.metrics.to_string(index=False))
    print("\nFigures:")
    for path in analysis.figure_paths:
        print(path)


if __name__ == "__main__":
    main()
