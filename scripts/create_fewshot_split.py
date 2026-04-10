import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List


def sorted_npy_files(path: Path) -> List[Path]:
    files = [p for p in path.iterdir() if p.is_file() and p.suffix == ".npy"]
    try:
        return sorted(files, key=lambda p: int(p.stem))
    except ValueError:
        return sorted(files, key=lambda p: p.name)


def remove_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic few-shot split for goal-reach dataset.")
    parser.add_argument("--src", type=str, required=True, help="Source dataset root with train/val/test")
    parser.add_argument("--out", type=str, required=True, help="Output dataset root")
    parser.add_argument("--train-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--train-link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
        help="How to materialize selected train files",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    train_src = src / "train"
    val_src = src / "val"
    test_src = src / "test"

    for required in (train_src, val_src, test_src):
        if not required.exists():
            raise FileNotFoundError(f"Missing required directory: {required}")

    if out.exists():
        if not args.force:
            raise FileExistsError(f"Output path already exists: {out}. Use --force to overwrite.")
        remove_if_exists(out)

    (out / "train").mkdir(parents=True, exist_ok=True)

    train_files = sorted_npy_files(train_src)
    if args.train_size > len(train_files):
        raise ValueError(f"Requested {args.train_size} train files, but only {len(train_files)} available.")

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(train_files, args.train_size), key=lambda p: p.name)
    selected_mapping = []
    for idx, src_file in enumerate(selected):
        dst_name = f"{idx}.npy"
        dst_file = out / "train" / dst_name
        if args.train_link_mode == "symlink":
            dst_file.symlink_to(src_file)
        else:
            shutil.copy2(src_file, dst_file)
        selected_mapping.append({"dst": dst_name, "src": src_file.name})

    # val/test are full evaluation sets, linked as directories.
    (out / "val").symlink_to(val_src, target_is_directory=True)
    (out / "test").symlink_to(test_src, target_is_directory=True)

    manifest = {
        "source_root": str(src),
        "output_root": str(out),
        "train_size": args.train_size,
        "seed": args.seed,
        "train_link_mode": args.train_link_mode,
        "selected_train_files": [p.name for p in selected],
        "selected_train_mapping": selected_mapping,
    }
    with (out / "fewshot_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
