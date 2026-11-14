# Failed-recording audit — November 14, 2026

The full preprocessing job 27554188 completed successfully in 3:22:34 with
64,973 successful decodes/extractions and 49 failed recordings out of 65,022.
All downstream cohorts, tables, report and notebook were generated. The notebook
now uses full audio scope; all 14 code cells executed without saved errors.
These 49 errors are distinct from the earlier native-worker crash incident.

## Findings

All 49 local files match the content size AND MD5 checksum returned by Synapse
for their authorized Voice Activity table file handles. Metadata were checked
through the existing authenticated file-handle service; no audio was downloaded
and no credentials or signed URLs were saved. Thus the local copies match the
source objects. Re-downloading those same objects is not a repair.

| Count | File structure | FFmpeg diagnostic |
|---:|---|---|
| 18 | Exactly 28 bytes: MP4 `ftyp` box only, with no movie metadata or audio payload | `moov atom not found` |
| 1 | 16,392 bytes: `ftyp` followed by an invalid/unrecognized zero-type box; no `moov` or `mdat` | `moov atom not found` |
| 30 | MP4/ALAC containers with only 8 or 16 bytes of `mdat` payload; headers advertise approximately 0.09 or 0.19 seconds | `invalid samples per frame: 0`, decode error rate 1 |

The damaged ALAC headers do not establish a valid recording duration. These
structures are consistent with empty/incomplete source captures; the exact
upstream recording or upload failure cannot be determined from the files.

A four-core read-only Slurm audit (job 27562918, COMPLETED, exit 0, six seconds)
replayed decoding for all 49 files and tested FFmpeg error-tolerant decoding
(`-err_detect ignore_err -fflags +discardcorrupt`). Both modes produced **zero
samples for every file**. No partial waveform is available to run through
Praat, openSMILE or spectral extraction. No source files or completed canonical
outputs were changed. The 49 recordings should retain explicit failure status;
missing acoustic values must not be replaced with zeros or synthetic audio.

The 49 failed recordings have 32 distinct SHA256 hashes. Some empty/malformed
objects therefore have identical bytes; this is not evidence of shared valid
voice recordings. Do not interpret these failed-file hashes as vocal similarity.

## Evidence and reproducibility

The versioned read-only diagnostic is `scripts/audit_audio_failures.py`:

```bash
# Run through a Slurm allocation, with approved paths configured.
python scripts/audit_audio_failures.py --root "$MPOWER_OUTPUT_ROOT" --workers 4
```

It audits failed entries from the full QC table and writes a timestamped private
directory under `eda/diagnostics/`. Evidence from this run is in:

```text
pd_data/mpower_phase1/eda/diagnostics/failed_audio_audit_20261114T065949Z/
  failures.csv                        per-recording file structure/decode summary
  <recording_id>.json                  MP4 box layout and complete decoder evidence
  summary.json                        aggregate decoder results
  source_checksum_comparison.json     per-file source/local checksum and size checks
  source_checksum_summary.json        aggregate source-match results
  verify_remote.py                    private metadata-verification script
```

Participant-bearing identifiers and paths stay in that private directory. The
three structural diagnostic tests passed, covering header-only files, payload
byte counts, zero/extended-size boxes and truncated boxes. No complete extraction
rerun is needed because none of the source files or extraction values changed.

Recommendation: retain these failures and their reasons in the audit trail.
Recovery would require a different valid upstream recording/object from the
data provider; changing decoder settings cannot recreate absent audio samples.
No provider contact or additional dataset download was performed.
