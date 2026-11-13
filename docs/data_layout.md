# mPower Data Architecture

## Source

- Project: syn4993293
- Table: syn5511444
- Task: Parkinson's voice analysis

## Storage

The dataset lives outside the Git repository.

Configure the root path with:

    MPOWER_DATA_ROOT

The expected structure is:

    $MPOWER_DATA_ROOT/
        synapse_cache/
        voice_activity/
            Voice_Activity.csv
            metadata_parts/
            audio/

## Metadata

The consolidated metadata file is:

    voice_activity/Voice_Activity.csv

The downloader adds an audio_path column to identify
the downloaded file belonging to each record.

The audio file-handle column is:

    audio_audio.m4a

The remaining columns originate from the Synapse table.

## Recordings

Audio files are stored under:

    voice_activity/audio/

File extensions may not always reflect audio format.
Use the metadata manifest to locate each recording.

## Resuming

The downloader creates metadata_parts/part_*.csv.

Each complete batch is verified before being skipped.

A failed or interrupted batch can be retried by
resubmitting the same Slurm job.

Keep the batch size unchanged between resumptions.

## Security

Raw participant metadata, audio, and access tokens
must not be committed to Git.

Use only institutionally approved data storage.

For the complete sibling `pd_env` / `pd_data` structure, three environment roots,
private output meanings and cloud access limitations, see [pd_data context](PD_DATA_CONTEXT.md).
