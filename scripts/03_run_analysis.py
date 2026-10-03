"""Run the full study and write every exhibit to outputs/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import run_all


def main() -> None:
    results = run_all()

    print("\n=== Granger: panel specification (primary) ===")
    print(results["granger"]["panel"].to_string(index=False))

    print("\n=== Granger: single-series ===")
    single = results["granger"]["single"]
    print(single.to_string(index=False) if len(single) else "(none estimable)")

    print("\n=== Media negativity by outlet bucket ===")
    print(results["bias"]["by_bucket"].to_string(index=False))

    if "by_region" in results["bias"]:
        print("\n=== Social sentiment by author region ===")
        print(results["bias"]["by_region"].to_string(index=False))

    print("\n=== Negativity on up vs down days ===")
    print(results["bias"]["asymmetry"].to_string(index=False))

    print("\n=== ADF diagnostics ===")
    print(results["granger"]["diagnostics"].to_string(index=False))


if __name__ == "__main__":
    main()
