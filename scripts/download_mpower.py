#!/usr/bin/env python3
"""Resumable mPower Voice Activity downloader for HPC."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import synapseclient

TABLE_ID = "syn5511444"
AUDIO_COL = "audio_audio.m4a"


def handle_id(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {"", "nan", "none", "null"}:
        return ""

    return text[:-2] if text.endswith(".0") else text


def batch_complete(part_path, expected):
    if not part_path.is_file():
        return False

    try:
        df = pd.read_csv(
            part_path,
            dtype=str,
            keep_default_na=False
        )

        if (
            len(df) != expected
            or "audio_path" not in df
            or AUDIO_COL not in df
        ):
            return False

        for fh, location in zip(
            df[AUDIO_COL],
            df["audio_path"]
        ):
            if not handle_id(fh):
                continue

            if (
                not location
                or not Path(location).is_file()
                or Path(location).stat().st_size == 0
            ):
                return False

        return True

    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        required=True,
        help="Approved HPC storage directory"
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=250
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 downloads the full table"
    )

    args = parser.parse_args()

    if args.chunk < 1 or args.limit < 0:
        parser.error(
            "--chunk must be >= 1 and --limit must be >= 0"
        )

    token = os.environ.get("SYNAPSE_AUTH_TOKEN")

    if not token:
        sys.exit(
            "Missing SYNAPSE_AUTH_TOKEN. "
            "Use the configured Slurm job."
        )

    root = Path(args.out).expanduser().resolve()

    voice = root / "voice_activity"
    audio = voice / "audio"
    parts = voice / "metadata_parts"
    cache = root / "synapse_cache"

    for directory in (audio, parts, cache):
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    syn = synapseclient.Synapse(
        cache_root_dir=str(cache),
        configPath=str(root / "synapse_no_config.ini")
    )

    syn.login(
        authToken=token,
        silent=True
    )

    total = int(
        syn.tableQuery(
            f"SELECT COUNT(*) FROM {TABLE_ID}"
        ).asDataFrame().iloc[0, 0]
    )

    if args.limit:
        total = min(total, args.limit)

    print(
        f"Voice Activity rows: {total:,}",
        flush=True
    )

    print(
        f"Batch size: {args.chunk}",
        flush=True
    )

    print(
        f"Storage directory: {root}",
        flush=True
    )

    expected_parts = []

    for offset in range(0, total, args.chunk):
        count = min(
            args.chunk,
            total - offset
        )

        part = parts / f"part_{offset:07d}.csv"

        expected_parts.append(part)

        if batch_complete(part, count):
            print(
                f"SKIP completed rows "
                f"{offset + 1:,}-{offset + count:,}",
                flush=True
            )
            continue

        print(
            f"Downloading rows "
            f"{offset + 1:,}-{offset + count:,}",
            flush=True
        )

        query = (
            f"SELECT * FROM {TABLE_ID} "
            f"ORDER BY ROW_ID ASC "
            f"LIMIT {count} OFFSET {offset}"
        )

        response = syn.tableQuery(query)
        frame = response.asDataFrame()

        if len(frame) != count:
            raise RuntimeError(
                f"Expected {count} rows, "
                f"received {len(frame)} "
                f"at offset {offset}"
            )

        downloaded = syn.downloadTableColumns(
            response,
            [AUDIO_COL],
            downloadLocation=str(audio)
        )

        lookup = {
            handle_id(k): str(Path(v).resolve())
            for k, v in downloaded.items()
        }

        frame["audio_path"] = [
            lookup.get(handle_id(fh), "")
            for fh in frame[AUDIO_COL]
        ]

        for fh, location in zip(
            frame[AUDIO_COL],
            frame["audio_path"]
        ):
            if not handle_id(fh):
                continue

            if (
                not location
                or not Path(location).is_file()
                or Path(location).stat().st_size == 0
            ):
                raise RuntimeError(
                    "Missing downloaded audio for "
                    f"file handle {handle_id(fh)}"
                )

        temporary = part.with_suffix(".csv.tmp")

        frame.to_csv(
            temporary,
            index=False
        )

        temporary.replace(part)

        print(
            f"Saved checkpoint: {part.name}",
            flush=True
        )

    print(
        "Combining completed metadata batches...",
        flush=True
    )

    if expected_parts:
        merged = pd.concat(
            [
                pd.read_csv(
                    p,
                    dtype=str,
                    keep_default_na=False
                )
                for p in expected_parts
            ],
            ignore_index=True
        )
    else:
        merged = pd.DataFrame()

    manifest = voice / "Voice_Activity.csv"
    temporary_manifest = voice / "Voice_Activity.csv.tmp"

    merged.to_csv(
        temporary_manifest,
        index=False
    )

    temporary_manifest.replace(manifest)

    print(
        f"DOWNLOAD COMPLETE: {len(merged):,} rows",
        flush=True
    )

    print(
        f"Metadata: {manifest}",
        flush=True
    )

    print(
        f"Audio directory: {audio}",
        flush=True
    )


if __name__ == "__main__":
    main()
