
## mPower Voice Dataset

### Project overview

This project develops machine-learning methods for
Parkinson's disease screening using speech recordings.

Data source:
- Synapse mPower Public Researcher Portal
- Project ID: syn4993293
- Voice Activity table: syn5511444

### Data storage

The real dataset is NOT stored in GitHub.

It is stored on approved persistent HPC storage.

The root directory is specified using:

    MPOWER_DATA_ROOT

Expected data layout:

    $MPOWER_DATA_ROOT/
        voice_activity/
            Voice_Activity.csv
            metadata_parts/
            audio/
        synapse_cache/

The primary audio column is:

    audio_audio.m4a

The metadata contains an additional column:

    audio_path

This column identifies the local HPC path to each
downloaded recording.

### Data access

Code must read the data path from the environment.

Example:

    import os
    from pathlib import Path

    root = Path(os.environ["MPOWER_DATA_ROOT"])

    metadata = root / "voice_activity" / "Voice_Activity.csv"

Do not assume data exists within the Git repository.

### Cloud coding agents

Cloud coding agents may not have direct access
to the private HPC filesystem.

When writing or testing code:

1. Use synthetic fixtures when necessary.
2. Read docs/data_layout.md for filesystem conventions.
3. Do not invent actual participant records.
4. Keep all real participant data outside Git.
5. Write scripts that can execute through Slurm.
6. Use environment variables rather than hardcoded paths.
7. Do not commit tokens, recordings, or raw participant metadata.

### ML development

Use participant-disjoint data splitting where appropriate
to avoid leakage between recordings from the same participant.

Keep preprocessing, training, and evaluation scripts
separate from data storage.

The dataset may only be accessed and used
under its applicable research-data permissions.

### HPC execution

The existing Conda environment is named:

    pd_env

Batch jobs use Slurm.

Download script:

    scripts/download_mpower.py

Slurm script:

    scripts/download_mpower.slurm

Do not download the full dataset onto a developer laptop.
