# Observed local mPower data dictionary

Source: `$MPOWER_DATA_ROOT/voice_activity/Voice_Activity.csv`. The observed consolidated export contains 65,022 rows and nine columns. These counts describe the inspected export, not the entire Synapse project.

| Observed column | Role | Linkage / interpretation |
|---|---|---|
| recordId | Recording key | No duplicates or missing values in inspected source; source rows retained |
| healthCode | Participant grouping key | 5,826 distinct values; repeated recordings expected |
| createdOn | Recording timestamp | Raw value retained in source; parse ISO or plausible epoch milliseconds; audit failures |
| appVersion | App version | Technical covariate, not a clinical label |
| phoneInfo | Device metadata | Technical covariate, not a participant key |
| audio_audio.m4a | Sustained-phonation file handle | Link through exported audio_path; do not infer identity from filenames |
| audio_countdown.m4a | Countdown handle | Existing downloader does not retrieve/map countdown audio; no local mapping inferred |
| medTimepoint | Medication timing response | 300 missing responses; cannot establish PD diagnosis or actual drug exposure |
| audio_path | Downloaded sustained recording path | All 65,022 mapped paths exist; downloaded names may end in .tmp |

The voice export alone has no diagnosis, age, gender, UPDRS or PDQ-8 fields. An explicitly authorized metadata-only download of Demographics Survey `syn5511429` is now available under `$MPOWER_DATA_ROOT/demographics/Demographics.csv` (6,805 rows, 31 columns). Its observed fields include `healthCode`, `recordId`, `createdOn`, `professional-diagnosis`, `age`, `gender`, `diagnosis-year`, `medication-start-year`, `education`, `race`, `smoked`, and `health-history`. Participant keys are unique and nonmissing. The table matches 5,044/5,826 voice participants and 63,255/65,022 recordings. Professional diagnosis values are exactly `True`/`False`/missing; interpretation is participant-reported professional diagnosis, not independent clinical adjudication. Age is baseline age; gender is the survey’s reported gender and is not inferred biological sex. No severity surveys were downloaded. Do not infer healthy status from medication responses. Additional discovered tables are schema-audited and kept separate unless linkage is explicitly verified. Download shards are compared with consolidated rows and are never appended as new observations.

The private `table_schemas.json` includes dimensions, storage dtypes, numeric parse fractions, missingness, distinct values (up to 100 categories), candidate key duplicates and first five rows. This file contains participant data and must remain outside Git. The private inventory covers raw storage and repository files separately; mapped audio is recognized regardless of suffix. Extension-based discovery alone cannot classify arbitrary unmapped .tmp files as recordings.

Canonical recording output has a unique analysis row hash, source table/CSV row number, original recordId, participant ID, handle, path, task, timestamp, medication, diagnosis and identity flags. Duplicate source rows remain individually traceable and ineligible for primary release. Participant output summarizes recording count, label consistency, age/sex availability/conflicts and follow-up. Age median is descriptive; longitudinal age variation is flagged and not interpreted as an error automatically.

Feature names use `praat_full__`, `praat_stable__`, `egemaps__` and `spectral__` namespaces. `processed/feature_dictionary.csv` documents interval, units, implementation, sample rate and validity for emitted columns. Full eGeMAPS descriptor names are preserved; consult its descriptor definitions for feature-specific units.

The audited demographic join preserves unmatched voice participants and does not multiply or delete source recordings. Demographic source columns use a `demo__` namespace. `source_diagnosis` retains the direct professional-diagnosis mapping; `diagnosis` additionally quarantines controls with contradictory diagnosis-year, medication-year or explicit pre/post-PD-medication responses. The signed baseline-survey-to-recording time difference is retained. See `demographic_join_audit.json`, the missing-participant lists, and `diagnosis_medication_audit.csv` in private output storage.
