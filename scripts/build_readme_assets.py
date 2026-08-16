"""Generate the tracked public README and repository-preview assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_dynamics.reporting.readme_assets import build_readme_assets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "assets" / "readme",
        help="Directory for generated public assets.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=ROOT / "reports" / "tables",
        help="Directory containing the frozen authoritative evidence tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_readme_assets(args.output_dir.resolve(), args.tables_dir.resolve())
    for path in outputs:
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
