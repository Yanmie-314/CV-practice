from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], step_name: str, cwd: str | None = None) -> int:
    """Run a subprocess command and stream output to terminal in real-time."""
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Start: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    sys.stdout.flush()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
    )
    # Stream output line by line to terminal
    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()

    process.wait()
    ret = process.returncode
    print(f"\n{'='*60}")
    if ret == 0:
        print(f"  STEP COMPLETE: {step_name}")
    else:
        print(f"  STEP FAILED: {step_name} (exit code {ret})")
    print(f"  End: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    sys.stdout.flush()
    return ret


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full adaptive RGB-T experiment pipeline.")
    parser.add_argument("--conda-env", default="yolo_gaiic", help="Conda environment name")
    parser.add_argument("--fusion", default="adaptive", choices=["rgbt", "tir_rgb_gray", "weighted", "adaptive"])
    parser.add_argument("--model", default="yolo11s.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int, default=None, help="Smoke test limit")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip dataset preparation if already done")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if already done")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-test", action="store_true", help="Skip test set inference")
    args = parser.parse_args()

    env = args.conda_env
    fusion = args.fusion
    out_dir = f"datasets/gaiic_yolo_rgbt_{fusion}"
    yaml_path = f"configs/gaiic_yolo_rgbt_{fusion}.yaml"
    project_name = f"yolo11s_rgbt_{fusion}"
    work_dir = f"work_dirs/{project_name}"
    best_pt = f"{work_dir}/weights/best.pt"

    total_start = time.time()

    # =====================================================================
    # Step 1: Dataset Preparation
    # =====================================================================
    if not args.skip_data_prep:
        ret = run_command(
            [
                "conda", "run", "-n", env, "python",
                "scripts/prepare_rgbt_yolo_dataset_v2.py",
                "--fusion", fusion,
                "--out-dir", out_dir,
                "--yaml-out", yaml_path,
                "--overwrite",
                "--verbose",
                "--progress-interval", "100",
            ] + (["--max-images", str(args.max_images)] if args.max_images else []),
            step_name=f"1. Prepare {fusion.upper()} RGB-T dataset",
        )
        if ret != 0:
            print("\n!!! Dataset preparation failed. Aborting.")
            sys.exit(1)
    else:
        print(f"\n[SKIP] Dataset preparation (using existing {out_dir})")

    # =====================================================================
    # Step 2: Training
    # =====================================================================
    if not args.skip_train:
        ret = run_command(
            [
                "conda", "run", "-n", env, "yolo", "detect", "train",
                f"model={args.model}",
                f"data={yaml_path}",
                f"imgsz={args.imgsz}",
                f"epochs={args.epochs}",
                f"batch={args.batch}",
                f"device={args.device}",
                "project=work_dirs",
                f"name={project_name}",
            ],
            step_name=f"2. Train YOLO11s on {fusion.upper()} dataset",
        )
        if ret != 0:
            print("\n!!! Training failed. Aborting.")
            sys.exit(1)
    else:
        print(f"\n[SKIP] Training (using existing {best_pt})")

    # Check best.pt exists
    if not Path(best_pt).exists():
        print(f"\n!!! Model not found: {best_pt}. Aborting.")
        sys.exit(1)

    # =====================================================================
    # Step 3: Validation
    # =====================================================================
    if not args.skip_eval:
        val_pred = f"submissions/{fusion}_val.json"
        ret = run_command(
            [
                "conda", "run", "-n", env, "python",
                "scripts/predict_rgbt_to_coco.py",
                "--model", best_pt,
                "--source", f"{out_dir}/images/val",
                "--out", val_pred,
                "--split", "val",
                "--conf", "0.001",
            ],
            step_name="3. Validation inference",
        )
        if ret == 0:
            run_command(
                [
                    "conda", "run", "-n", env, "python",
                    "scripts/eval_coco_results.py",
                    "--pred", val_pred,
                ],
                step_name="3b. Validation evaluation (mAP)",
            )
    else:
        print("\n[SKIP] Validation evaluation")

    # =====================================================================
    # Step 4: Test Set Inference
    # =====================================================================
    if not args.skip_test:
        test_pred = f"submissions/{fusion}_test.json"
        final_sub = f"submissions/final_submission_{fusion}.json"
        ret = run_command(
            [
                "conda", "run", "-n", env, "python",
                "scripts/predict_rgbt_to_coco.py",
                "--model", best_pt,
                "--source", f"{out_dir}/images/test",
                "--out", test_pred,
                "--split", "test",
                "--conf", "0.001",
            ],
            step_name="4. Test set inference",
        )
        if ret == 0:
            run_command(
                [
                    "conda", "run", "-n", env, "python",
                    "scripts/make_submission.py",
                    "--input", test_pred,
                    "--out", final_sub,
                ],
                step_name="4b. Generate final submission",
            )
    else:
        print("\n[SKIP] Test set inference")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Fusion mode: {fusion}")
    print(f"  Model: {best_pt}")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
