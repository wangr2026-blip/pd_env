# Pretrained checkpoints and sustained-vowel inputs

For exact commands to reproduce the downloads on another machine or through
Slurm, start with [Download pretrained models](DOWNLOAD_PRETRAINED_MODELS.md).

The public checkpoint inventory is in [pretrained_models.json](../configs/pretrained_models.json), with exact upstream revisions, asset sizes and integrity identifiers in [pretrained_models.lock.json](../configs/pretrained_models.lock.json). On November 13, 2026, this local checkout downloaded and integrity-verified **nine public checkpoints, 66 files, totaling 12,018,426,907 bytes (12.02 GB)** under Git-ignored `models/`. Consult `models/download_status.json` for recorded outcomes and timestamps. **Downloaded and integrity-verified does not mean loaded, integrated or evaluated, or available on HPC.** HeAR access returned HTTP 401 and is pending by user choice. The commands below exclude it. Model assets remain local and are excluded from Git commits.

The [downloader](../scripts/download_pretrained_models.py) needs Python 3 and curl. It selects one PyTorch weight format, preferring safetensors, and includes frontend/configuration assets. Local layout:

```text
models/
  download_status.json
  checkpoints/<alias>/
    config.json
    preprocessor_config.json
    model.safetensors or pytorch_model.bin
    ...support files...
    download_provenance.json
```

Each provenance file records the pinned revision and downloaded-file SHA-256 hashes. Verification compares asset sizes and upstream LFS SHA-256 or Git blob hashes. Incomplete transfers use `.part` files; repeating the command resumes transfers and verifies existing assets. From the repository root, these commands select all nine public checkpoints while leaving HeAR pending:

```bash
public_models=(ast_audioset wav2vec2_base_960h wav2vec2_large_robust
               wavlm_base_plus wavlm_large hubert_large
               whisper_small whisper_medium whisper_large_v3)

python3 scripts/download_pretrained_models.py --models "${public_models[@]}" --plan-only
python3 scripts/download_pretrained_models.py --models "${public_models[@]}"
python3 scripts/download_pretrained_models.py --models "${public_models[@]}" --verify-only
python3 -m json.tool models/download_status.json
```

The default command without `--models` or `--public-only` also attempts HeAR and returns a nonzero exit status if its gated files remain inaccessible. Use the explicit public aliases above while HeAR is pending. The existing lock is reused unless `--refresh-lock` is explicitly requested. Avoid refreshing it during a reproducible run. On HPC, set `MPOWER_MODEL_ROOT` to an approved external model directory or pass `--output-root`; then inspect the status file under that root. Public weights can be transferred there without moving participant data. Recordings, metadata and derived embeddings remain outside Git under the [existing data conventions](data_layout.md).

After verification, models can be loaded from local checkpoint directories. This standalone example demonstrates loading assets; it does not integrate the expanded inventory into the benchmark runner:

```python
import os
from pathlib import Path
from transformers import AutoFeatureExtractor, WhisperModel

root = Path(os.environ.get("MPOWER_MODEL_ROOT", "models"))
checkpoint = root / "checkpoints" / "whisper_small"
frontend = AutoFeatureExtractor.from_pretrained(checkpoint, local_files_only=True)
encoder = WhisperModel.from_pretrained(checkpoint, local_files_only=True).get_encoder()
encoder.requires_grad_(False).eval()
```

`HF_HUB_OFFLINE=1` can additionally prevent Hub access. Pin the installed Transformers/PyTorch versions and record the model class and preprocessing configuration alongside experiments.

**A roughly ten-second M4A recording should retain its duration and pitch.** Decode to a finite float32 waveform, inspect the true sample rate/channels/duration, convert to mono and correctly resample to 16 kHz. A ten-second mono input then contains 160,000 samples. M4A is a container/codec detail; a WAV file need not be written. Relabeling a sample rate changes playback speed, while genuine resampling preserves timing. Do not squeeze ten seconds into two, repeat a short vowel or use one saved spectrogram image for every architecture. The existing FFmpeg decoder in [mpower_common.py](../scripts/mpower_common.py) already produces mono float32 audio at the requested rate.

| Checkpoints | Native frontend and shape | Handling a ten-second vowel |
|---|---|---|
| wav2vec2, WavLM, HuBERT | 16 kHz mono waveform, `[batch, samples]` | Encode the full 160,000 samples; use each checkpoint's normalization and masking rules. |
| AST AudioSet | 128 Mel bins, 1,024 time frames: `[batch, 1024, 128]` | Approximately its natural ten-second range; the frontend pads/truncates to its configured length. |
| Whisper Small/Medium | Native 30-second frontend: `[batch, 80, 3000]` | Pad the ten-second input to the native window; retain acoustic encoder representations. |
| Whisper Large-v3 | Native 30-second frontend: `[batch, 128, 3000]` | Same duration policy, but its own 128-bin processor is required. |
| HeAR, when accessible | Two seconds: `[batch, 32000]`; preprocessing gives `[batch, 1, 192, 128]`; 512-dimensional pooled output | Divide an exact ten seconds into five adjacent two-second windows, retaining all five embeddings before aggregation. |

These shapes follow the [AST frontend](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/blob/main/preprocessor_config.json), [Whisper Medium frontend](https://huggingface.co/openai/whisper-medium/blob/main/preprocessor_config.json), [Large-v3 frontend](https://huggingface.co/openai/whisper-large-v3/blob/main/preprocessor_config.json) and [official HeAR preprocessing](https://github.com/Google-Health/hear/blob/master/python/data_processing/audio_utils.py).

Normalization is checkpoint-specific. WavLM Base+ specifies `do_normalize=false`, while WavLM Large specifies `true`. AST uses its pretrained AudioSet statistics. Preserve native frontends instead of applying universal waveform standardization. Base wav2vec2's group-normalized frontend has different encoder-mask guidance from layer-normalized variants such as Robust; do not carry the original base assumption into every waveform model. Exclude padded positions from downstream pooling using actual output-length calculations. For AST and Whisper, cropping padded output positions does not undo attention to padded regions earlier in the forward pass. ([WavLM Base+ frontend](https://huggingface.co/microsoft/wavlm-base-plus/raw/main/preprocessor_config.json), [WavLM Large frontend](https://huggingface.co/microsoft/wavlm-large/blob/main/preprocessor_config.json), [wav2vec2 documentation](https://huggingface.co/docs/transformers/model_doc/wav2vec2))

HeAR's two-second window is a valid input constraint, rather than evidence that the model is unsuitable for sustained phonation. Its health-audio training also does not prove a Parkinson advantage. Mean or mean-plus-standard-deviation pooling across all five embeddings is a proposed baseline, not a validated Google Parkinson protocol. Establish one policy for short recordings and residual tails; padded samples provide no extra evidence. Official preprocessing uses min/max operations over its input tensor, so initially preprocess windows individually before batching and verify batch invariance. ([HeAR model card](https://developers.google.com/health-ai-developer-foundations/hear/model-card), [preprocessing source](https://github.com/Google-Health/hear/blob/master/python/data_processing/audio_utils.py))

An ordered experiment plan is:

1. Decode/QC once and use the same accepted span, participants and labels for every model. Keep all recordings and windows from each participant in one split.
2. Compare full-duration representations: the complete ten seconds for speech models and AST/Whisper, versus all five HeAR windows pooled. Start with frozen encoders and a regularized classifier; tune layers and pooling inside training folds.
3. Run a matched-context comparison in which every model sees the identical five two-second windows with comparable recording-level aggregation. Native padding remains architecture-specific. This matches source audio exposure, while computation and token counts still differ.
4. Compare Whisper Small, Medium and Large-v3 using the same splits, classifier family, layer-selection budget and metrics. Report encoder-only memory/time and parameter counts. This is a practical checkpoint comparison: Large-v3 also changes pretraining data and Mel-bin count, so it does not isolate model size. An optional original-generation Small/Medium/Large-v1 comparison would reduce those confounds. ([Large-v3 model card](https://huggingface.co/openai/whisper-large-v3))
5. Retain a conventional phonation-feature baseline and demographic controls. Report participant-level AUROC/AUPRC, sensitivity at a fixed specificity and participant-level confidence intervals. Choose thresholds and models without consulting the final test set.

Local inspection of the downloaded safetensors headers counted 88,154,112 encoder tensor elements for Whisper Small, 307,216,384 for Medium and 636,968,960 for Large-v3. The stored encoder tensors are F32 in Small/Medium and F16 in Large-v3. These are checkpoint-header observations, not measured runtime memory or speed. Keep inference precision consistent when comparing sizes; similar on-disk file sizes do not imply similar capacity or compute.

The expanded inventory is now integrated through
[the local benchmark configuration](../configs/frozen_benchmark_local.yaml).
The runner supports all nine public checkpoints and loads local pinned assets;
WavLM/HuBERT use checkpoint-native normalization and masking. See the
[local integration guide](LOCAL_PRETRAINED_BENCHMARK.md) for commands and
validation scope. The original three-model/five-second configuration remains
available explicitly. HeAR integration remains excluded pending access.
