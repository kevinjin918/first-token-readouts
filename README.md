# When first-token readouts misread feature steering

Code, result files and paper source for

> Kevin Jin. **When First-Token Readouts Misread Feature Steering in a Medical Vision–Language
> Model.** The Third Workshop on GenAI for Health, NeurIPS 2026.

Interventions on vision–language models are often scored on yes/no questions by the first output
token. In MedGemma 1.5 (4B), suppressing the most active features of a cross-layer transcoder on
chest radiographs changes the answers the model writes, while the first-token readout barely moves.
This repository holds the code for every experiment in the paper, the result file behind every
number, and the script that regenerates the paper's numbers, tables and figures from those files.

**Transcoder weights.** 34 layers, 2,048 features per layer, full cross-layer span, trained on
MedGemma 1.5-4b-it.

Zenodo, [10.5281/zenodo.22105454](https://doi.org/10.5281/zenodo.22105454) (CC BY 4.0):
`clt_ckpt.pt`, the training checkpoint (6.24B allocated parameters, 3.30B of them trained, plus
optimizer state; 37,630,140,293 bytes, MD5 `4c2747c75e4ce368bd48ccd194f9b8ef`).

`CrossLayerTranscoder.load_checkpoint` reads the checkpoint directly.
[`scripts/export_clt_safetensors.py`](scripts/export_clt_safetensors.py) optionally writes the
trained parameters to a weights-only `.safetensors` file (no pickle), checks every tensor is
bitwise equal, and `load_checkpoint` reads that file too.

## Reproduce the paper from the result files (CPU, minutes)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis,viz,dev]"
python paper/make_numbers.py      # rewrites paper/numbers.tex and paper/tables/ from results/
git diff --stat paper/            # empty: every number in the paper regenerates
bash paper/build.sh               # numbers, figures and the PDF (needs tectonic)
pytest -q
```

Every number in `paper/main.tex` is a `\val{key}` defined in `paper/numbers.tex`, which
`paper/make_numbers.py` writes from `results/`; no number is typed by hand. CI repeats the
regeneration on every push and fails if any number changes.

## Rerun the experiments (one 80 GB GPU)

```bash
pip install -e ".[models,data,analysis]"
python scripts/download_nih.py --sample      # NIH ChestX-ray14 images_001: 4,999 images, 1,335 patients
mkdir -p ckpt && curl -L -o ckpt/clt_ckpt.pt \
    "https://zenodo.org/records/22105454/files/clt_ckpt.pt?download=1"   # 37.6 GB
python scripts/readout_sampled_answers.py --ckpt ckpt/clt_ckpt.pt \
    --out results/clt_scale/readout_sampled_answers.json
```

MedGemma is gated on Hugging Face: accept its terms and log in (`hf auth login`) first. The steering
scripts share the arguments `--ckpt`, `--data-dir` (default `~/.cache/tracecxr/data/chestxray14`),
`--results` (the result files they build on) and `--out`. Each script's docstring states what it
tests and, for the pre-registered tests, the decision rule; Appendix A of the paper lists every
rule, its outcome and every amendment. Retraining the transcoder (`scripts/clt_live.py`, defaults as
released) needs the full ChestX-ray14 set (`--full`) and the CheXpert Plus reports (see Data).

## Where each result comes from

| Paper | Script | Result file (`results/clt_scale/`) |
|---|---|---|
| §3, Figure 2a,b, occlusion as the reference effect | `occlusion_readout_audit.py` | `occlusion_readout_audit.json` |
| §4, §5, suppression and the written answers, false and correct calls (E4) | `readout_sampled_answers.py` | `readout_sampled_answers.json` |
| §4.1, App. B, the two routes; parsing | `answer_text_checks.py`, `parse_audit_sheet.py` | `answer_text_checks.json`, `hand_reading_scored.json` |
| §4.2, App. D, the edit on the live model (E8, E9) | `clamp_live_mlp.py`, `clamp_live_dose.py` | `clamp_live_mlp.json`, `clamp_live_dose.json` |
| §3, Figure 2c, evidence locations for other findings | `causal_map_general.py`, `attn_causal_map.py` | `causal_map_*.json`, `attn_causal_map.json` |
| Not in the paper: evaluation without an intervention (E6, E7); rules and outcomes in the docstrings and result files | `readout_eval_answers.py`, `readout_eval_matched.py`, `readout_eval_matched_power.py`, `readout_eval_resolution.py`, `readout_prompt_confound.py` | `readout_eval_*.json`, `readout_prompt_confound.json` |
| §5, App. E, donor sets (E3, E5) | `donor_sets_patient_disjoint.py`, `clamp_contrastive_answers.py`, `clamp_amplify_donor_specificity.py` | `donor_sets_patient_disjoint.json`, `clamp_contrastive_answers.json`, `clamp_amplify_donor_specificity.json` |
| App. C, the transcoder | `clt_live.py` (the released run), `clt_scale.py` and `clt_cache.py` (cached and per-layer baselines), `clt_replacement.py`, `clt_fair_eval.py`, `clt_fvu_by_prompt.py`, `clt_metric_transfer.py`, `feats_causal_vs_inert.py`, `stage1_train_transcoder.py` | `clt_live_result.json`, `clt_scale_result.json`, `clt_span0_result.json`, `replacement_live.json`, `clt_replacement*.json`, `fair_eval.json`, `clt_fvu_by_prompt.json`, `clt_prompt_battery.json`, `feats_causal_vs_inert_norm.json`, `../a1_fidelity.json` |
| §4, App. C, the first check and reproducing the suppression (E2) | `readout_generation_check.py` | `readout_generation_check.json` |

`results/clt_scale/logs/` holds the generation runs' logs, which the paper's compute times come from.

## Layout

- `tracecxr/`: the library. `models/` wraps MedGemma and CheXagent and computes the first-token
  readouts (`models/_base.py`); `transcoder/` trains, evaluates and exports the cross-layer
  transcoder; `intervention/` holds the clamp (`clamp.py`), the sampled-answer runner and the
  answer parser (`generation_check.py`).
- `scripts/`: one script per experiment, plus `download_nih.py` and the transcoder export.
- `results/`: the result files.
- `paper/`: the LaTeX source, `make_numbers.py`, the figure scripts and `build.sh`.
- `tests/`: unit tests on mock models and tiny transcoders; no GPU, weights or data needed.

## Data

The experiments use NIH ChestX-ray14 (Wang et al., CVPR 2017), which is public and not redistributed
here; the result files identify radiographs by their ChestX-ray14 file names. The transcoder was
trained on MedGemma's activations for ChestX-ray14 radiographs and, as separate inputs, for report
texts from CheXpert Plus (Chambon et al., 2024), which is distributed by Stanford AIMI under a
research use agreement. No report text is included here.

## Licence

Code: Apache 2.0 (`LICENSE`). Result files and paper text: CC BY 4.0. Transcoder weights: CC BY 4.0.
MedGemma is subject to the Health AI Developer Foundations terms of use.

## Citation

```bibtex
@inproceedings{jin2026firsttoken,
  title     = {When First-Token Readouts Misread Feature Steering in a Medical
               Vision--Language Model},
  author    = {Jin, Kevin},
  booktitle = {The Third Workshop on GenAI for Health: Agentic Systems, Clinical Trust,
               and Future Potential (NeurIPS 2026 Workshop)},
  year      = {2026}
}
```
