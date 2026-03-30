"""
main.py - Orchestrator: run all pipeline phases end-to-end.

Usage:
    python main.py [--phase {1,2,3,4,5,all}]

Phases:
    1 – Train baseline
    2 – Generate adversarial examples
    3 – Train detectors
    4 – Adversarial training + denoising defense
    5 – Evaluate & report
"""

import argparse
import sys

import config
from utils import set_seed


def run_phase1():
    print("\n" + "=" * 70)
    print("PHASE 1: Baseline CNN Training")
    print("=" * 70)
    import train
    train.main()


def run_phase2():
    print("\n" + "=" * 70)
    print("PHASE 2: Adversarial Attack Generation")
    print("=" * 70)
    if not config.BASELINE_MODEL_PATH.exists():
        print("[main] Baseline model not found - running Phase 1 first.")
        run_phase1()
    import attacks
    attacks.main()


def run_phase3():
    print("\n" + "=" * 70)
    print("PHASE 3: Detection Mechanisms")
    print("=" * 70)
    if not (config.ADV_DIR / "pgd" / "adv_tensors.pt").exists():
        print("[main] Adversarial tensors not found - running Phase 2 first.")
        run_phase2()
    import detect
    detect.main()


def run_phase4():
    print("\n" + "=" * 70)
    print("PHASE 4: Defenses")
    print("=" * 70)
    if not config.BASELINE_MODEL_PATH.exists():
        run_phase1()
    import defense
    defense.main()


def run_phase5():
    print("\n" + "=" * 70)
    print("PHASE 5: Evaluation & Report")
    print("=" * 70)
    import evaluate
    evaluate.main()


PHASE_MAP = {
    "1": run_phase1,
    "2": run_phase2,
    "3": run_phase3,
    "4": run_phase4,
    "5": run_phase5,
}


def main():
    set_seed()
    parser = argparse.ArgumentParser(
        description="Adversarial Attacks in Healthcare AI - end-to-end pipeline"
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "4", "5", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    args = parser.parse_args()

    if args.phase == "all":
        for fn in PHASE_MAP.values():
            fn()
    else:
        PHASE_MAP[args.phase]()

    print("\n[OK] Pipeline complete. Check metrics/ and plots/ for outputs.")


if __name__ == "__main__":
    main()
