"""
Rewrite parquet files produced by build_parquet.py with small row groups.

build_parquet.py writes each split as ONE row group (e.g. train.parquet
is a single ~8 GB group). Parquet can only be read a whole row group at
a time, so training from such a file means every sample access (and
every DataLoader worker) tries to load the entire split into memory.
This script copies the same rows into a new file with a few rows per
group, so training reads a few images at a time. Image bytes are copied
unchanged; nothing is decoded or re-encoded.

Run once after build_parquet.py, then train from the new directory:

    python src/repartition_parquet.py --src data/parquet --dst data/parquet_rg16

    from data_loading import official_splits_parquet
    train_ds, hold_ds, test_ds = official_splits_parquet("data/parquet_rg16")

The originals are left untouched. Once the new files check out, delete
the old ones to reclaim disk space (you need room for both while this
runs). Pair this with RowGroupShuffleSampler in training so shuffling
stays cheap.
"""

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def repartition(src: Path, dst: Path, rows_per_group: int = 16) -> int:
    """Copy src to dst with at most rows_per_group rows per row group.
    Returns the number of row groups written. Writes to a temporary file
    first so an interrupted run never leaves a half-written dst behind."""
    src, dst = Path(src), Path(dst)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    pf = pq.ParquetFile(src)
    n_groups = 0
    with pq.ParquetWriter(tmp, pf.schema_arrow, compression="snappy") as writer:
        # iter_batches streams a batch at a time instead of materializing the file
        for batch in pf.iter_batches(batch_size=rows_per_group):
            writer.write_table(pa.Table.from_batches([batch]))  # one small table = one row group
            n_groups += 1
    os.replace(tmp, dst)
    return n_groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="directory containing the *.parquet files")
    ap.add_argument("--dst", type=Path, required=True, help="directory to write the new files to")
    ap.add_argument("--rows-per-group", type=int, default=16)
    args = ap.parse_args()

    if args.src.resolve() == args.dst.resolve():
        raise SystemExit("--dst must be a different directory from --src")
    files = sorted(args.src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No .parquet files in {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)

    for f in files:
        n = repartition(f, args.dst / f.name, args.rows_per_group)
        before = pq.ParquetFile(f).metadata
        after = pq.ParquetFile(args.dst / f.name).metadata
        assert before.num_rows == after.num_rows, f"row count changed for {f.name}"
        print(f"{f.name}: {before.num_rows} rows, {before.num_row_groups} -> {after.num_row_groups} row groups")


if __name__ == "__main__":
    main()
