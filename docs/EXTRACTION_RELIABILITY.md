# Acoustic extraction repair and HPC validation — November 13, 2026

All 184 requested core measurements were already implemented: 44 Praat, 88
openSMILE eGeMAPSv02 Functionals and 52 spectral summaries. They are retained;
12 pitch/trim sensitivity values remain pilot-only. See ACOUSTIC_FEATURES.md.
No new pretrained-model experiment is implied by acoustic extraction.

## Confirmed failure and repair

The previous full job's error log contains over 32,000 native segmentation-fault
reports. Repeated tracebacks terminate in Librosa zero_crossing_rate ->
zero_crossings -> Numba's compiled gufunc, called by extract_acoustic_features.py.
A native segfault kills the worker and cannot be caught as a Python exception.
The prior recovery path then retried pending recordings serially, repeatedly
hitting the same crash and consuming the 24-hour allocation.

The replacement computes the same zero-crossing statistic with NumPy prefix
sums, preserving edge padding, frame boundaries, the 1e-10 threshold and the
positive-zero convention. Tests matched Librosa exactly for float32/float64,
random audio, silence, short inputs, near-zero values and odd/even frame sizes.
This removes the native gufunc from the production extraction path.

The old gufunc succeeded in a separate interpreter with a fresh cache on the
validation node. The precise binary/cache/environment trigger of the old crash
has therefore not been established. Node-local, job-specific Numba caches now
avoid reusing compiled artifacts across heterogeneous compute nodes; this is
an additional isolation measure, not proof that shared caching caused the crash.

The openSMILE frontend is reused within each worker instead of rebuilt for
every recording. Native BLAS/OpenMP threads remain limited to one per worker.
Exceptional native-crash retries run in parallel independent processes. If at
least half a batch (minimum two rows) still suffers native worker failures,
the runner saves a checkpoint and stops rather than processing the entire
census with a broken environment. Individual file errors remain explicit.
Existing code/config/version hashes invalidate incompatible extraction caches.

The feature dictionary also now identifies derived NHR as dimensionless and
spectral contrast as dB. Original recordings and prior pilot outputs were not
modified by validation.

## Compute-node validation

Slurm job 27553976 completed successfully on c-11-20, with 16 allocated CPUs,
48 GB requested RAM, elapsed 1:57 and reported batch peak RSS about 1.2 GiB.
That RSS is a scheduler-reported measurement for this small validation job,
not a production memory bound.

Each extraction processed the same 32 real recordings, including 24 recordings
marked worker_error in the failed job and eight from the successful pilot.
Every run decoded 32/32 and completed feature extraction for 32/32.

| Workers | Elapsed seconds | Recordings/second |
|---:|---:|---:|
| 1 | 26.10 | 1.23 |
| 4 | 8.98 | 3.56 |
| 8 | 7.98 | 4.01 |
| 16 | 5.93 | 5.39 |

The 16-worker measurement is 4.4 times faster than the single-worker run.
These are small-batch wall times including process startup; the first run can
also pay cold import/compilation/filesystem-cache costs. They do not establish
linear scaling or the full-census completion time. More cores beyond 16 have
not been benchmarked. Sixteen workers are the validated production setting.

Numeric outputs agreed across worker counts (rtol 1e-6, atol 1e-8). All 184 core
columns agreed with prior successful pilot results for eight shared recordings
(rtol 1e-5, atol 1e-7). A cached repeat hit all 32 records and took 1.45 seconds.
The full preprocessing regression suite passed 13 tests on the compute node,
including synthetic pipeline/report generation, native-crash containment,
feature coverage, ZCR equivalence and systematic-failure checkpoint/stop.
Fourteen warnings were Matplotlib/Pyparsing deprecation notices.

Private manifests, per-recording results, diagnostic code and logs are under
pd_data/extraction_validation_20261113/. The failed full-run checkpoint,
selection and diagnostics were copied to
pd_data/mpower_phase1/archive/full_failed_27498642/ before restarting.

## Runtime limit and production retry

The CPU partition defq advertises MaxTime=45 days. An actual --time=0 submission
was accepted with TimeLimit=UNLIMITED but remained PENDING with
Reason=PartitionTimeLimit. The job was updated to 45-00:00:00, removing that
block. A test-only submission had accepted unlimited without exposing this
scheduling restriction; acceptance alone did not prove eligibility to run.

The full template now requests 16 CPUs, 48 GB and 45 days. Other clusters can
override runtime through SBATCH_TIMELIMIT. This removes the old 24-hour cap;
it does not bypass scheduler policy. Per-recording/subprocess timeouts remain
in place to contain pathological files and hangs.

Production retry job: 27554188, observed RUNNING on m-09-04 after the runtime
correction. Submitted template and logs are in pd_data/slurm/.
Its copied template records the original unlimited request; the scheduler job
record carries the subsequent 45-day correction. Submission/initial progress
is not evidence of full completion. Check squeue/sacct and the full-scope final
outputs before replacing the pilot overview or citing full-dataset features.
