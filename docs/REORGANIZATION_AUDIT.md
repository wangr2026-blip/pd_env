# Repository and HPC cleanup — November 13, 2026

`pd_env` is the GitHub source repository; `pd_data` is private persistent HPC
storage. Laptop and cloud agents can develop against synthetic fixtures using
only the repository. They cannot assume HPC data or Python environments exist.

## What moved

| Original checkout location | Private HPC location |
|---|---|
| `logs/` (including a 187.6 MB error log) | `pd_data/slurm/logs/legacy/` |
| `.mpower_phase1_environment.tar.gz` (341.4 MB) | `pd_data/environments/mpower_phase1_environment.tar.gz` |
| `.mpower_venv/` | `pd_data/environments/mpower_venv/` |
| `.benchmark_venv/` | `pd_data/environments/benchmark_venv/` |

Ignored local compatibility symlinks preserve previous environment and log
paths. They are not tracked or included in clones. The environments retain
host-specific dependencies and must be installed afresh on other machines.
No recordings, source metadata, derived participant tables or checkpoints were
moved into the repository. Historical private handoff documents/configs remain
with their evidence; the maintained project documentation is already in `docs/`.
Private small tables and logs remain outside Git regardless of file size.

The three one-off `instruction*.md` task files, including
`instructions_cleanup.md`, were deleted as requested. `AGENTS.md` remains the
maintained repository guide for HPC and remote agents.

## Slurm use

Templates remain versioned under `scripts/` so remote agents can edit them.
Submitted copies and runtime logs belong in `pd_data/slurm/`. Configure paths
using `.env.example` (copy to ignored `.env` and replace placeholders). Then:

```bash
source .env
bash scripts/submit_mpower.sh frozen_benchmark --smoke-test
# Other template names: download_mpower, download_mpower_demographics,
# preprocess_mpower, linked_pilot_mpower, full_mpower.
```

The helper creates a uniquely named template copy under
`$MPOWER_SLURM_ROOT/jobs/`, directs logs to `$MPOWER_SLURM_ROOT/logs/`, and sets
the working directory to the code checkout. It rejects storage inside the
repository. Slurm options can be set through standard `SBATCH_*` variables,
for example `SBATCH_GRES=gpu:1` and a site-appropriate `SBATCH_PARTITION`.
The copied template is preserved; Python source still comes from the checkout,
so avoid editing code used by an active job. Direct legacy `sbatch` commands
retain their original `logs/` defaults; use the helper on new clones.
No jobs were submitted or cancelled during this cleanup.

## Verified run history

Read-only `sacct` queries outside the network-restricted sandbox verified:

| Job | Task | State | Elapsed |
|---|---|---|---|
| 27445756 | Initial voice download | COMPLETED | 00:01:20 |
| 27446125 | Voice download | COMPLETED | 05:12:14 |
| 27497408 | Initial phase 1 | COMPLETED | 00:07:37 |
| 27497846 | Demographics download | COMPLETED | 00:00:10 |
| 27497956 | Linked pilot | COMPLETED | 00:03:29 |
| 27498284 | First full phase 1 attempt | FAILED | 00:00:34 |
| 27498642 | Full phase 1 retry | TIMEOUT | 1-00:00:11 |

`squeue` showed no active mPower jobs at inspection; the only listed job was an
unrelated interactive dashboard. Earlier reports of controller DNS failure
refer to sandbox access, not evidence that the scheduler itself was down.

The full retry's last logged progress was 31,040 / 65,022 recordings. Its
preserved checkpoint contains 31,040 attempted rows: **30,990 worker failures,
41 other errors, and nine successful rows**. Native-worker failures dominate;
previous tracebacks implicated Librosa/Numba extraction. This does not establish
that those recordings are poor quality. The full feature run is incomplete;
repair and validate the worker path before resuming. Keep the checkpoint as
failure evidence rather than treating it as a completed feature table.

Historical verification reports record 28 passing regression tests, including
three real-recording tests with tiny random backbones. The additional 19 passing
targeted checks overlap that suite. No requested pretrained weights loaded and
no pretrained baseline completed. Model downloads were not retried in this
cleanup. The fixed 56-person pilot and proposed weighting protocol remain
exploratory; weighting is documented but not fitted or implemented.

## Cleanup validation

Slurm shell syntax, mocked submission paths/arguments, rejection of in-repository
job storage, relocated Python imports, Git ignore rules and patch whitespace
were checked. Mock submissions do not contact the scheduler. No model training
or full dataset reprocessing was performed for this storage/documentation change.
