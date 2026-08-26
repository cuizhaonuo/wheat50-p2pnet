"""Build a P2PNet-compatible dataset from a ShanghaiTech-.mat source.

Source layout (read-only), e.g. wheat50/instances_256x1024_min_area_rect_v2/:
  SRC_ROOT/
    train_data/{images,ground_truth}/
    val_data/{images,ground_truth}/
  images/<stem>.jpg
  ground_truth/GT_<stem>.mat   # mat["image_info"] is an (N, 2) array of "x y"
                               # (.h5 density maps in the same dir are ignored)

Output layout (created fresh):
  DST_ROOT/
    train_data/{images(symlinks), annotations(*.txt "x y" per line)}/
    val_data/{images, annotations}/
    shanghai_tech_part_a_train.list   # lines: "img_relpath gt_relpath"
    shanghai_tech_part_a_test.list

The two .list filenames are hardcoded in crowd_datasets/SHHA/SHHA.py, so we
keep those names regardless of the actual dataset.
"""
import argparse
import os
import shutil

import numpy as np
import scipy.io as io


def extract_points(mat_path):
    info = io.loadmat(mat_path)["image_info"]
    # This source stores image_info directly as an (N, 2) float array.
    # Fall back to the classic ShanghaiTech nesting if needed.
    pts = np.asarray(info)
    if pts.dtype == object or pts.ndim != 2 or pts.shape[1] != 2:
        pts = info[0][0][0][0][0]
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def find_mat(gt_dir, stem):
    for cand in (f"GT_{stem}.mat", f"{stem}.mat"):
        p = os.path.join(gt_dir, cand)
        if os.path.isfile(p):
            return p
    return None


def build_split(src_root, dst_root, split_dir):
    src_img_dir = os.path.join(src_root, split_dir, "images")
    src_gt_dir = os.path.join(src_root, split_dir, "ground_truth")
    dst_img_dir = os.path.join(dst_root, split_dir, "images")
    dst_ann_dir = os.path.join(dst_root, split_dir, "annotations")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_ann_dir, exist_ok=True)

    entries = []
    n_empty = 0
    for img_name in sorted(os.listdir(src_img_dir)):
        stem, ext = os.path.splitext(img_name)
        if ext.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        mat_path = find_mat(src_gt_dir, stem)
        if mat_path is None:
            print(f"[skip] missing gt mat for: {img_name}")
            continue

        # symlink image (idempotent: remove stale link first)
        src_img = os.path.abspath(os.path.join(src_img_dir, img_name))
        dst_img = os.path.join(dst_img_dir, img_name)
        if os.path.islink(dst_img) or os.path.exists(dst_img):
            os.remove(dst_img)
        os.symlink(src_img, dst_img)

        # write annotation txt
        points = extract_points(mat_path)
        if len(points) == 0:
            n_empty += 1
        txt_path = os.path.join(dst_ann_dir, stem + ".txt")
        with open(txt_path, "w") as f:
            for x, y in points:
                f.write(f"{x:.2f} {y:.2f}\n")

        img_rel = os.path.relpath(dst_img, dst_root)
        gt_rel = os.path.relpath(txt_path, dst_root)
        entries.append((img_rel, gt_rel))

    return entries, n_empty


def write_list(path, entries):
    with open(path, "w") as f:
        for img_rel, gt_rel in entries:
            f.write(f"{img_rel} {gt_rel}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True,
                    help="Source dataset (ShanghaiTech-.mat style) root.")
    ap.add_argument("--dst_root", required=True,
                    help="Output P2PNet dataset root. Will be (re)created.")
    ap.add_argument("--clean", action="store_true",
                    help="Remove dst_root before building.")
    args = ap.parse_args()

    if args.clean and os.path.isdir(args.dst_root):
        print(f"[clean] removing {args.dst_root}")
        shutil.rmtree(args.dst_root)
    os.makedirs(args.dst_root, exist_ok=True)

    train_entries, train_empty = build_split(args.src_root, args.dst_root, "train_data")
    val_entries, val_empty = build_split(args.src_root, args.dst_root, "val_data")

    write_list(os.path.join(args.dst_root, "shanghai_tech_part_a_train.list"), train_entries)
    write_list(os.path.join(args.dst_root, "shanghai_tech_part_a_test.list"), val_entries)

    print(f"train: {len(train_entries)} samples ({train_empty} empty)")
    print(f"val:   {len(val_entries)} samples ({val_empty} empty)")
    print(f"output: {args.dst_root}")


if __name__ == "__main__":
    main()
