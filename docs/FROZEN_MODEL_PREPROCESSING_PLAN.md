# Sustained-vowel preprocessing and frozen-model plan

Implementation update: the frozen-development stages are implemented in the
[execution guide](FROZEN_TESTING_IMPLEMENTATION.md), with immutable development
snapshots and dependent Slurm experiments. The original proposal/audit below
remains historical design context. Fine-tuning, validated task/near-duplicate
release review and final sealed-test evaluation remain later gates.

Yes: first hold each pretrained encoder fixed, train comparable classifier heads, select promising pipelines on development participants, then fine-tune a prespecified shortlist. This document distinguishes the implemented integration from remaining preprocessing/evaluation work. Source inspection reflects repository snapshot `84519b7` on November 14, 2026; private HPC artifacts were not accessed in this audit.

The latest repository evidence **does establish completed full acoustic preprocessing**: job 27554188 finished in 3:22:34, with 64,973 successful decodes/extractions among 65,022 source recordings and 49 failures. Cohorts, reports and the notebook were generated. A subsequent audit found all 49 failed objects matched Synapse sizes/MD5 checksums and yielded zero samples under normal and tolerant decoding. Retain their failure status and missing features; fetching the same source object cannot restore absent audio. This supersedes the earlier running-job snapshot and native-worker incident. It does not establish pretrained-model inference, validated QC cutoffs or final release splits. ([Current results](../docs/CURRENT_RESULTS.md#L8), [failed-file audit](../docs/FAILED_AUDIO_AUDIT.md#L1))

Completed cohort A contains 62,123 recordings from 4,854 people; B contains 61,651 from 4,830; matched sensitivity cohort C contains 22,820 from 838. These are candidate full-cohort resources for the proposed comparison, subject to the QC/split review below. The historical 56-person snapshot remains an execution pilot. Preserve immutable manifest/hash snapshots for new experiments; do not overwrite inputs being consumed by active HPC jobs.

**Nine public pretrained checkpoints are now integrated and numerically smoke-verified offline on HPC.** Job 27563234 completed in 2:11 on one A100 80GB; each checkpoint processed the same six real participant-disjoint recordings and completed logistic and MLP heads. Thirty-three distinct checks passed. This establishes execution of the pretrained models, not full-cohort training or validated predictive performance; the aggregate summary explicitly marks both as false. HeAR remains excluded pending access. ([Integration guide](../docs/LOCAL_PRETRAINED_BENCHMARK.md), [aggregate evidence](../docs/results/pretrained_integration_summary.json))

| Area | Implemented behavior and evidence | Proposed change or interpretation |
|---|---|---|
| Decode | FFmpeg produces finite mono float32 at 16 kHz; empty/overlong output is rejected. [Decoder](../scripts/mpower_common.py#L100) | Preserve mechanism; retain source/channel QC and explicit stage outcomes. |
| Provenance | Original media header, file hash, native PCM hash and mono16k PCM hash. [Inspection](../scripts/inspect_audio.py#L105) | Add native decoded sample counts and per-channel QC. `normalized_pcm` means rate/channel conversion, not loudness normalization. |
| Signal processing | No manual amplitude normalization, denoising, preemphasis or VAD trimming. [Decode/QC](../scripts/inspect_audio.py#L112) | Keep signal-preserving baseline; apply only each checkpoint's native frontend. |
| QC | Energy, clipping proxy, spectral and timing measures use the converted waveform. [Quality](../scripts/audio_quality.py#L6) | Also inspect native channels before conversion; distinguish technical faults from dysphonia. |
| Legacy SNR | `estimated_snr_db` equals the frame-RMS P90/P10 dynamic range. [Calculation](../scripts/audio_quality.py#L17) | Rename as energy dynamic range; do not use it as environmental SNR or a quality weight. |
| Cohort selection | B excludes less than two seconds of detected pitch; pitch coverage/F0 variability are review flags. [Criteria](../scripts/build_cohorts.py#L27) | Use a technical primary cohort and separate pitch-based sensitivity arm; measure diagnosis-dependent retention. |
| Persistence | EDA caches JSON features; frozen extraction separately writes waveform segment NPZs and selected tokens. [EDA](../scripts/inspect_audio.py#L99), [segments](../src/models/frozen.py#L147) | Successful EDA does not imply a saved preprocessed-WAV archive. Prefer pooled-feature caching for large frozen comparisons. |
| Model adapters | Architecture registry covers AST/Whisper/wav2vec2/WavLM/HuBERT, with no tokenizer requirement. Local aliases validate provenance and load offline. [Local loading](../src/models/frozen.py#L22), [dispatch](../src/models/frozen.py#L44) | Extend named feature taps, native AST pooling and eventually HeAR; retain implemented local loading. |
| Masks/lengths | Waveform singleton forwards, masks for layer-norm variants, convolution-derived lengths; Whisper pooling uses frontend mask counts. [Forward](../src/models/frozen.py#L65) | Validate any future batching changes; distinguish output cropping from encoder padding attention. |
| Windowing | Default local config uses ten-second windows/hops, potentially multiple chunks; tails below two seconds are omitted. Legacy config retains five-second windows. [Current config](../configs/frozen_benchmark_local.yaml#L16), [segmentation](../src/data/benchmark.py#L137) | Implement the shared first-ten-second span and endpoint-anchored two-second condition below. |
| Heads/evaluation | Train-only scaler; recording-row logistic/MLP losses; participant evaluation. [Heads](../src/training/heads.py#L10) | Add participant-balanced optimization and validation-only selection. Current `comparison.csv` contains test metrics. [Runner](../scripts/run_frozen_benchmark.py#L99) |

The components are conventional research tools, but the current thresholds do not constitute a universally validated clinical standard. Praat pitch/jitter/shimmer depend on analysis bounds and sample selection; undefined periodicity measures should stay missing. The implementation's longest pitch-detected interval is a sensitivity measurement, not proof of a stationary clinical segment. ([Acoustic extraction](../scripts/extract_acoustic_features.py#L41), [Praat guidance](https://www.fon.hum.uva.nl/praat/manual/Voice_6__Automating_voice_analysis_with_a_script.html))

For reproducibility, an authorized HPC operator can check accounting and aggregate artifacts below. These are reference checks, not commands executed by this audit. Use the approved environment with `MPOWER_OUTPUT_ROOT` set; never print participant rows or private paths.

```bash
sacct -j 27554188,27562918 --format=JobID,State,ExitCode,Elapsed,Start,End -P
```

```python
# REFERENCE: aggregate-only artifact coverage; run on approved HPC storage.
import json, os
from pathlib import Path
import pandas as pd
r = Path(os.environ["MPOWER_OUTPUT_ROOT"])
m = json.loads((r/"eda/diagnostics/full_execution.json").read_text())
print({k: m.get(k) for k in ("scope", "stage", "selected")})
source = pd.read_parquet(r/"processed/recording_manifest.parquet")
selected = pd.read_parquet(r/"processed/full_selection.parquet")
expected = set(selected.recording_id)
print("source/selected rows", len(source), len(selected))
print("selection coverage", len(set(source.recording_id)-expected))
for name in ("full_audio_quality_metrics.parquet", "full_acoustic_features.parquet"):
    f = pd.read_parquet(r/"processed"/name)
    seen = set(f.recording_id)
    print(name, {"rows": len(f), "unique": len(seen),
                 "missing": len(expected-seen), "unexpected": len(seen-expected)})
    for col in ("decode_status", "feature_status"):
        if col in f:
            s = f[col].astype("string").fillna("missing")
            counts = {v: int(s.eq(v).sum()) for v in
                      ("ok", "partial", "error", "worker_error", "not_requested", "missing")}
            counts["other"] = len(s)-sum(counts.values())
            print(col, counts)
```

Reconcile `source_metadata_sha256`, `config_digest`, `script_sha256` and `versions` in `full_execution.json` against the intended submitted run, plus dates and scheduler outcomes. Audit feature-family `status__*` counts and undefined measures separately. The `full_features_checkpoint.parquet` is progress evidence, distinct from final tables. Confirm cohort/split/report provenance references those final outputs. A stage marked `ok` means execution succeeded, not every jitter value exists; current `decode_status` also covers probe/hash/QC exceptions, motivating finer stage statuses. ([Execution metadata](../scripts/inspect_audio.py#L149), [final tables](../scripts/inspect_audio.py#L198))

The proposed common pipeline is: preserve the original object; probe/hash; decode native channels once for QC; derive one immutable mono16k float32 waveform; QC the derived signal; select a common source span; run native model frontends. Cache private arrays under approved environment-configured roots with source hashes, decoder/backend versions and transform settings. Record native rate, channels, sample count and duration, then derived sample count. Lossy-codec bit depth can be unavailable. Correct resampling preserves time/pitch; merely relabeling the rate does not. No waveform plots or spectrogram PNGs are required for model input. ([FFmpeg output controls](https://ffmpeg.org/ffmpeg.html))

Native-channel QC should inspect clipping/flat-top runs, dropouts, DC, amplitude and channel disagreement before downmixing; stereo cancellation can be hidden by mono-only inspection. Also retain analysis-rate RMS/peak/silence proxies and decode warnings. Full-scale samples after lossy decoding/resampling do not uniquely establish microphone clipping. Environmental SNR is unavailable unless a defensible noise reference exists; HNR is harmonicity, a different quantity. Do not normalize away amplitude cues in the acoustic branch simply because an encoder normalizes its input.

The primary cohort should require resolved identity/label, usable finite nonempty audio and task compliance, with at least two real seconds as a declared **research feasibility rule**, not a clinical cutoff. Establish compliance through a validated review/detector process: the current task label derives from the source column. Flag quietness, breathiness, breaks and aperiodicity rather than automatically treating them as bad audio. Pitch/jitter failure need not invalidate spectral or learned representations. Cohort B's `pitch_detected_s >= 2` can preferentially remove pathological voices; use it as a sensitivity analysis. Coverage below 50%, F0 variability, clipping, low amplitude and HNR/jitter availability are not current cohort-B exclusion criteria. Report retention by diagnosis, demographics, device and medication where available. Denoising, loudness normalization and adaptive stable-window selection belong in prespecified secondary experiments.

For the primary span, use `[0,min(N,160000))` at 16 kHz without adaptive edge trimming. The ten-second cap can remove the original utterance offset; record whether and how much audio was capped. A full-utterance or stable-center ablation must have a separately prespecified span/window policy shared across models. Reject primary spans below 32,000 real samples with an explicit reason. For two-second models, use endpoint-anchored complete windows:

```python
# REFERENCE: proposed common window policy, not deployed code.
import numpy as np
span = waveform[:160000]
if len(span) < 32000:
    raise ValueError("below_2s_primary_minimum")
k = (len(span)+31999)//32000
starts = np.rint(np.linspace(0, len(span)-32000, k)).astype(int)
windows = [span[s:s+32000] for s in starts]
assert starts[0] == 0 and starts[-1]+32000 == len(span)
assert np.all(np.diff(starts) <= 32000)
```

Exactly ten seconds yields five adjacent windows. At 9.7 seconds, five windows overlap by 75 ms between neighbors and cover the entire span without a fabricated tail. Just over two seconds yields two heavily overlapping windows; record their redundancy. Log unique covered duration, summed window duration and overlap. One recording produces one feature vector; overlapping windows are correlated inputs, never independent participants. Equal-duration window embeddings receive uniform aggregation weights. One-window population standard deviation is zero.

Run two explicit conditions: full-context models encode the entire span while HeAR aggregates every window; the context-controlled condition sends the identical two-second windows to every model before aggregation. Both expose models to the same source samples, while native padding/context and compute still differ. Do not squeeze, repeat or independently select the best excerpt. Duration/overlap strata and optional central-span analyses are secondary, label-independent policies.

The current registry loads pinned local snapshots from `MPOWER_MODEL_ROOT/checkpoints/<alias>` using `local_files_only=True`, validates repository/revision provenance and fails explicitly on missing local assets. Numerical smoke execution is established for all nine public checkpoints, separately from byte verification; HeAR remains pending. Extend existing metadata with the proposed context policy, named feature taps and pooling, retaining checkpoint/frontend revisions, software/backend/dtype, normalization and valid lengths. No tokenizer or transcript generation is needed.

| Adapter | Class / native input | Representation and normalization |
|---|---|---|
| wav2vec2 Base-960h | `Wav2Vec2Model`, `[B,T]` | Native waveform normalization; group-normalized convolution; begin unpadded singleton forwards, no encoder attention mask for this checkpoint. |
| wav2vec2 Robust | `Wav2Vec2Model`, `[B,T]` | Native normalization; layer-normalized convolution; retain its returned attention mask. |
| WavLM Base+/Large | `WavLMModel`, `[B,T]` | Base+ `do_normalize=false`, group norm; Large `true`, layer norm. Both snapshots return masks. |
| HuBERT Large | `HubertModel`, `[B,T]` | Native normalization and mask; layer-normalized convolution. |
| AST | `ASTModel`, `[B,1024,128]` | Native fbank `(x-mean)/(2*std)`; primary 768D `pooler_output`, averaging final-normalized CLS/distillation tokens. |
| Whisper | `WhisperModel(...).get_encoder()` | Small/Medium `[B,80,3000]`, Large-v3 `[B,128,3000]`; native log-Mel, 30-second padding, explicit `do_normalize=false`. |
| HeAR, pending | `AutoModel` plus Google's preprocessing | Two-second `[B,32000]` to `[B,1,192,128]` Mel-PCEN; 512D `pooler_output`. |

Frontend settings follow the [locked Robust processor](https://huggingface.co/facebook/wav2vec2-large-robust/blob/2493a2c576276145c3e066d9243b0e391fab673a/preprocessor_config.json), [WavLM Base+](https://huggingface.co/microsoft/wavlm-base-plus/blob/4c66d4806a428f2e922ccfa1a962776e232d487b/preprocessor_config.json), [WavLM Large](https://huggingface.co/microsoft/wavlm-large/blob/c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c/preprocessor_config.json) and each local snapshot. Start float32, no augmentation, zero native dither where supported, then enable mixed precision only after numerical checks.

```python
# REFERENCE: illustrates the implemented class registry and frozen extraction.
# Additional feature/window policies in this document remain proposed.
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model, WavLMModel, HubertModel
classes = {"wav2vec2": Wav2Vec2Model, "wavlm": WavLMModel, "hubert": HubertModel}
feature = AutoFeatureExtractor.from_pretrained(snapshot, local_files_only=True)
encoder = classes[architecture].from_pretrained(snapshot, local_files_only=True)
encoder.requires_grad_(False).eval().to(device)
b = feature([span], sampling_rate=16000, padding=False, return_tensors="pt")
args = {k: v.to(device) for k,v in b.items() if k in {"input_values","attention_mask"}}
with torch.inference_mode():
    out = encoder(**args, output_hidden_states=True, return_dict=True)
```

Waveform group normalization depends on temporal padding. The runner now uses true-length singleton forwards for all waveform families, passes encoder masks to layer-normalized variants and omits them for group-normalized variants. WavLM Base+'s processor declares a mask, but its current unpadded singleton path contains no artificial waveform padding; validate native-mask semantics before changing batching. Convolution-derived valid lengths are implemented: `L=(L-kernel)//stride+1` through the configured stack gives 499 tokens for ten seconds and 99 for two. Mean/statistical pooling exists. Proposed named taps must distinguish `conv_raw` (512 channels), normalized pre-projection features and `hidden_states[0]` (projected, model-width encoder input); the latter is not raw convolution output. HuBERT needs an explicit convolution tap because its output lacks `extract_features`. Name transformer taps by resolved depth. ([Current lengths](../src/models/frozen.py#L109), [wav2vec2 source v4.50.3](https://github.com/huggingface/transformers/blob/v4.50.3/src/transformers/models/wav2vec2/modeling_wav2vec2.py))

AST's native pooler is the primary representation; the current wrapper instead discards special tokens and pools patches from selected hidden states. In inspected Transformers v4.50.3, `hidden_states[-1]` precedes final layer normalization, while `last_hidden_state`/pooler include it. Name patch/statistical pooling as a separate ablation, respecting the frequency/time patch grid. Native frontend padding occurs before normalization. ([AST model](https://github.com/huggingface/transformers/blob/v4.50.3/src/transformers/models/audio_spectrogram_transformer/modeling_audio_spectrogram_transformer.py), [frontend](https://github.com/huggingface/transformers/blob/v4.50.3/src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py))

Whisper uses each checkpoint's own 80/128-bin frontend and native frame count. The current runner requests frontend masks and derives real pooling lengths from them: approximately 500 positions for ten seconds, 100 for two. In inspected upstream v4.50.3, encoder `attention_mask` is accepted but ignored; AST likewise lacks encoder padding masking there. Excluding trailing output positions does not undo earlier padding attention. This source tag is a reference audit, not the established HPC runtime version. Exact executed package versions are stored in private run provenance but absent from the reviewed public summary; retrieve aggregate version values before locking the expanded protocol. Small/Medium/Large-v3 comparisons also change pretraining and frontend, so describe checkpoint performance rather than a pure capacity scaling law. ([Current Whisper path](../src/models/frozen.py#L71), [reference encoder](https://github.com/huggingface/transformers/blob/v4.50.3/src/transformers/models/whisper/modeling_whisper.py))

```python
# REFERENCE: Whisper encoder loaded from its own local snapshot/frontend.
b = feature([span], sampling_rate=16000, return_tensors="pt",
            padding="max_length", truncation=True,
            return_attention_mask=True, do_normalize=False)
assert b.input_features.shape[-1] == 3000
with torch.inference_mode():
    out = encoder(input_features=b.input_features.to(device),
                  attention_mask=b.attention_mask.to(device), return_dict=True)
n_valid = min((int(b.attention_mask[0].sum())+1)//2, out.last_hidden_state.shape[1])
tokens = out.last_hidden_state[0, :n_valid].float()
```

For HeAR, preprocess each cloned window independently: current Google min/max operations span the input batch and preprocessing can mutate tensors. Stack processed spectrograms, obtain `[K,512]` pooler outputs, then compute equal-weight mean (512D) or mean plus population standard deviation (1024D). Pin Google's preprocessing revision and test batch independence. This is proposed multiwindow pooling, not a clinically validated HeAR protocol. The downloader currently omits `.py` assets: if the release requires custom `AutoModel` code, review and pin/cache those files plus preprocessing sources/dependencies before enabling `trust_remote_code` or claiming offline readiness. Weights/configuration alone do not establish a runnable HeAR adapter. ([Official preprocessing](https://github.com/Google-Health/hear/blob/master/python/data_processing/audio_utils.py), [official quick start](https://github.com/Google-Health/hear/blob/master/notebooks/quick_start_with_hugging_face_pytorch.ipynb))

```python
# REFERENCE: after HeAR access and model loading are validated.
spec = torch.cat([preprocess_audio(torch.tensor(w)[None].clone()) for w in windows])
hear.requires_grad_(False).eval()
with torch.inference_mode():
    z = hear(spec.to(device), return_dict=True).pooler_output.float()
embedding = torch.cat([z.mean(0), z.std(0, correction=0)])  # 1024D
```

For practical full-cohort frozen runs, cache pooled recording features by default; make token/all-layer caches opt-in on bounded subsets. At 65,000 recordings, roughly 500 tokens, width 1,024 and float32, one token layer occupies approximately 133 GB per model before compression, versus about 266 MB for one pooled 1,024D vector per recording. Actual lengths and counts differ. The current runner persists selected token activations and requests all hidden states; selective layer capture and pooled-cache defaults remain proposed optimizations. Cache identities must include checkpoint/source/span/frontend/pooling settings and software versions. ([Extraction/cache](../src/models/frozen.py#L117))

The initial trained model is a train-fold feature scaler plus L2 logistic regression, with a small declared regularization search. Compare a small MLP next; learned attention/layer mixing is a separately labeled trained pooling experiment. Frozen encoders use `requires_grad_(False)`, `eval()` and inference-only offline extraction. For online head training, use `no_grad()` or clone features outside inference mode into ordinary tensors; do not feed inference tensors directly into a head that needs autograd. Cached NumPy features converted into fresh tensors avoid that restriction. Keep the encoder in eval mode even when a composite classifier enters training mode. Fit scalers, PCA, attention and all learned transforms only on training folds. ([Current heads](../src/training/heads.py#L10), [sklearn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html), [PyTorch inference mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html))

The following reference illustrates participant-balanced logistic fitting. `Xtrain` has one row per recording; arrays, labels and participant IDs come from the verified fold manifest. These are conceptual inputs, not a production function. Repeat the fit for each development candidate `chosen_C`, then select C using participant-aggregated validation predictions only.

```python
# REFERENCE: training/validation inputs only; no test-label access.
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
_, inverse, counts = np.unique(train_participant_ids,
                               return_inverse=True, return_counts=True)
w = 1.0 / counts[inverse]
w /= w.mean()
scaler = StandardScaler().fit(Xtrain, sample_weight=w)
head = LogisticRegression(C=chosen_C, max_iter=1000)
head.fit(scaler.transform(Xtrain), ytrain, sample_weight=w)
val_prob = head.predict_proba(scaler.transform(Xval))[:, 1]
# Aggregate val_prob by validation participant before comparing candidate C values.
# Select any operating threshold separately using validation predictions, never test.
```

Define the primary outcome as participant-level screening at an initial assessment, using one initial eligible recording or a fixed acquisition budget. All visits/windows and duplicate-connected people stay within one fold. Repeated-record training should weight each recording proportional to `1/n_recordings_for_participant`, recomputed within the training fold; normalize weights consistently and use matching weighted scaler moments/loss. If directly training windows, divide further by within-recording window count or average window loss first. Current training is unweighted per recording, despite participant-level evaluation. Day-balanced longitudinal analysis is separate. Weighted sklearn MLP support needs a pinned version at least 1.7 or a weighted PyTorch head; the repository's broader dependency range does not guarantee that API. ([MLP API](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html))

Use the same people and manifest digests for all encoders; disclose extraction failures rather than silently changing cohorts. With sufficient data, use development-only duplicate-aware group CV and a fixed final holdout. The existing 56-person pilot is for execution; eight validation people cannot support precise ranking or high-specificity calibration. Audit age/device-only baselines, demographic common support and medication subgroups; keep medication text out of primary voice-only classification. Exact duplicate grouping exists, but near-duplicate discovery and selection-bias review remain release prerequisites. ([Split checks](../scripts/create_participant_splits.py#L21), [current cohort status](../docs/CURRENT_RESULTS.md#L37))

**Do not shortlist using the existing `comparison.csv`: it contains participant test metrics.** Add a development mode that never reads test labels or exports test scores. Rank complete pipelines using a predeclared participant validation metric and budget; select the top two affordable backbones, with tie/seed handling fixed beforehand. Frozen rankings need not persist after adaptation. Keep the final test sealed throughout frozen selection, fine-tuning, calibration and threshold selection. Current binary metrics use 0.5; the proposed operating threshold is learned from participant-level validation/out-of-fold predictions for a declared specificity target, then carried unchanged to test. ([Test export](../scripts/run_frozen_benchmark.py#L112), [current metrics](../src/evaluation/metrics.py#L6), [threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html))

Fine-tuning needs a separate differentiable audio path: warm up the head, then try final encoder blocks or validated LoRA adapters under a limited development search. Remove inference/no-grad contexts around trainable blocks; cached final embeddings cannot update encoder weights. Verify optimizer membership, nonzero intended gradients, changing unfrozen weights and unchanged frozen weights. Use new adapted-state/cache signatures; retain original frozen artifacts. Predeclare the final frozen comparator and adapted systems before opening test. Report participant AUROC/AUPRC, achieved sensitivity/specificity at the carried threshold, calibration, coverage and paired participant/component-bootstrap intervals. Never treat windows as independent observations. ([Transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html))

| Milestone | Acceptance evidence before advancing |
|---|---|
| Audit/protocol | Reconcile full-run artifacts; release duplicate-aware membership; record exclusions, overlap burden and common acquisition policy. |
| Decode/QC/window implementation | Synthetic M4A, mono/stereo 16/44.1/48 kHz; duration/pitch preservation; native clipping/cancellation; silence/irregular phonation; 1.999/2/2.001/9.7/10/>10-second boundaries; exact coverage without fabricated tails. |
| Adapter extensions | Nine public pretrained smoke runs already passed. Extend tests for proposed native AST pooling, named convolution/layer taps and matched windows; audit load warnings and verify finite outputs/batch behavior. HeAR remains optional until accessible. |
| Frozen/head runner | Encoder state unchanged, head state changes; train-only moments; duplicating identical visits does not increase participant contribution; cache invalidates on relevant changes. |
| Selection/fine-tuning | Validation-only leaderboard; test-label perturbation cannot change fit/shortlist/threshold; intended fine-tune gradients verified; models/seeds/threshold locked. |
| Final evaluation | Same untouched participants; predeclared systems; aggregate uncertainty/coverage reports; no test-driven reruns presented as confirmatory. |

All pseudocode above is reference design; class dispatch, local loading and frozen singleton inference are already implemented, while new span/window, representation, balancing and selection policies remain proposals. This documentation update performs no pipeline implementation, HPC data processing or training. Keep private arrays, manifests, embeddings and predictions outside Git, run workloads through maintained Slurm templates and use environment-configured storage. Completed acoustic extraction and nine pretrained smoke checks supply the foundation for a leakage-controlled full-cohort research comparison.
