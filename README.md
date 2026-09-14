# Domain-Balanced Conditional Diffusion for MIMII DUE

This project is being converted in place from the DCASE 2021 dense-AE
baseline into a diffusion-based anomalous sound detector. No separate AE copy
is maintained here.

## Target pipeline

```text
WAV
  -> 128-bin log-FBank
  -> 128 x 128 spectrogram patches
  -> domain/section-conditioned diffusion U-Net
  -> partial DDIM reconstruction
  -> residual-based patch scores
  -> audio-level anomaly score
  -> source/target AUC and pAUC
```

Training uses normal sounds only. Source and target patches will be balanced,
and condition dropout will provide an unknown-domain path.

## Project layout

```text
diffusion/
  config.py          YAML loading and validation
  records.py         shared audio and patch metadata
  dataset.py         dataset indexing, FBank extraction, and patching
  conditioning.py    section/domain IDs and condition dropout
  unet.py            timestep-conditioned 2D denoising U-Net
  diffusion.py       DDPM training and DDIM reconstruction
  scoring.py         residual and audio-level anomaly scores
  metrics.py         DCASE-compatible AUC and pAUC
  engine.py          training and evaluation orchestration
00_train.py          training entry point
01_test.py           evaluation entry point
diffusion.yaml       experiment configuration
tests/               fast structural and unit tests
```

## Implementation stages

1. Project scaffold and configuration - complete.
2. File-level dataset indexing and 128 x 128 FBank patches - complete.
3. Unconditional DDPM training baseline - complete.
4. Partial DDIM reconstruction, anomaly scoring, and CSV evaluation - complete.
5. Domain-balanced section/domain conditioning.

The original dense-AE source files and its generated ``model/``/``result/``
artifacts have been removed, as this directory now belongs to the diffusion
experiment.

## Reproducible environment

The non-PyTorch runtime dependencies are pinned exactly in
``requirements.txt``. PyTorch is intentionally excluded so installing this
project never replaces the server's existing CUDA-specific build. The
selected NumPy 2.0.2 also satisfies ``opencv-python-headless==4.13.0.92`` when
OpenCV is already installed in a shared server environment.

```bash
python -m pip install --upgrade --no-cache-dir -r requirements.txt
python -m pip check
python -c "import torch, numpy, scipy, sklearn, yaml; print(torch.__version__, numpy.__version__, torch.cuda.is_available())"
```

The project does not depend on OpenCV, TensorFlow, ONNX, or protobuf, so those
packages are intentionally not managed by this requirements file.

## Validate the data pipeline

Run a quick local or server-side check without training a model:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
python 00_train.py --check-data --machine-type fan --max-files 6
```

Add ``--build-cache`` to write float16 log-FBank features under
``feature_cache/``. Repeat ``--machine-type`` to check several machines; omit
it to inventory all configured machine types.

## Train the stage-3 unconditional DDPM

The project trains one model per machine type. Before a full run, verify the
complete GPU path with one real fan batch:

```bash
python 00_train.py --smoke-test --machine-type fan
```

This uses the full 42.3M-parameter U-Net with batch size 1, performs one
forward/backward/optimizer/EMA update, and writes
``checkpoints/smoke/fan/last.pt``. It does not produce anomaly scores yet.

Start full training for one machine, or omit ``--machine-type`` to train all
five machines sequentially:

```bash
python 00_train.py --machine-type fan
python 00_train.py
```

Training uses batch size 8 with three-step gradient accumulation (effective
batch size 24), mixed precision on CUDA, domain-balanced sampling, EMA, and
float16 feature caching. Each machine keeps only ``last.pt`` for resuming and
the smaller ``ema.pt`` for evaluation; periodic full-state snapshots are not
created. Resume the latest epoch-level checkpoint with:

```bash
python 00_train.py --machine-type fan --resume
```

For a short diagnostic run, set an optimizer-step limit such as
``--max-steps 10``.

## Evaluate with partial DDIM reconstruction

First run the end-to-end smoke evaluation. It selects one normal and one
anomalous file from every fan section/domain, uses non-overlapping patches,
and performs five DDIM steps:

```bash
python 01_test.py --smoke-test --machine-type fan
```

The smoke metrics contain only one file per class and therefore verify the
pipeline only; they are not meaningful experiment results. A useful server
benchmark before the full run is:

```bash
python 01_test.py --machine-type fan --max-files-per-group 5
```

Run the complete fan development-set evaluation with:

```bash
python 01_test.py --machine-type fan
```

The default evaluation keeps the dense five-frame test stride and uses 15
deterministic DDIM denoising steps from timestep 280 (``ddim_stride=20``).
Runtime/quality ablations can be launched without editing YAML, for example:

```bash
python 01_test.py --machine-type fan --ddim-stride 10
python 01_test.py --machine-type fan --test-patch-hop 32 --batch-size 64
```

Outputs are written under ``outputs/``: six anomaly-score CSV files per
machine, a grouped ``metrics.csv``, and ``summary.json`` containing arithmetic
and harmonic means of AUC and pAUC.

## Diagnose fan scoring without retraining

The fan sweep reuses the trained EMA checkpoint. For every DDIM start step it
reconstructs each patch once, then evaluates absolute and ReLU anomaly filters,
multiple pixel TopK ratios, and several patch-to-audio aggregations in memory.
Start with a pipeline check:

```bash
CUDA_VISIBLE_DEVICES=1 python 01_test.py --fan-sweep --smoke-test
```

Then run the full default sweep:

```bash
CUDA_VISIBLE_DEVICES=1 python 01_test.py --fan-sweep --machine-type fan
```

The default start steps are ``100 200 280 400``. Step 600 is intentionally
left for a second pass because it substantially increases DDIM runtime. Custom
values can be supplied without changing the YAML:

```bash
CUDA_VISIBLE_DEVICES=1 python 01_test.py --fan-sweep \
  --sweep-start-steps 100 200 280 400 600 \
  --sweep-topk-ratios 0.01 0.03 0.05 0.1 0.2 1.0 \
  --sweep-aggregations mean max topk_mean \
  --patch-topk-ratio 0.1
```

Results are stored in ``outputs/fan_sweep/``. The summary table is sorted by
the harmonic mean over all six AUCs and six pAUCs, while the group table keeps
every section/domain result. ``fan_sweep_best.json`` contains the highest
development-set setting. Since labelled development-test data selects that
setting, freeze it before evaluating other machines or final evaluation data.

## Evaluate fan with residual-distribution GMMs

This stage keeps the trained fan U-Net fixed. It reconstructs normal training
audio and test audio from timestep 400, pools each residual map over time into
a 128-dimensional frequency vector, and averages patch vectors into one vector
per audio file. Two-component GMM negative log likelihood is then evaluated for
signed, absolute, and ReLU residuals using global or section-specific models and
full or diagonal covariance matrices.

Run a small end-to-end check first:

```bash
CUDA_VISIBLE_DEVICES=1 python 01_test.py --fan-gmm --smoke-test
```

Run the complete fan experiment with:

```bash
CUDA_VISIBLE_DEVICES=1 python 01_test.py --fan-gmm --machine-type fan
```

The default patch hop is 32 for both normal fitting data and test data. This
keeps the two residual distributions comparable while avoiding the large amount
of redundant computation caused by a five-frame hop. It can be overridden with
``--gmm-patch-hop``. Outputs are written to ``outputs/fan_gmm/``:

- ``fan_gmm_summary.csv`` ranks all GMM configurations;
- ``fan_gmm_groups.csv`` contains every section/domain metric;
- ``fan_gmm_audio_scores.csv`` preserves file-level scores for later analysis;
- ``fan_residual_features.npz`` preserves file-level residual vectors so new
  statistical scorers can be tested without repeating DDIM reconstruction;
- ``fan_gmm_best.json`` and ``fan_gmm_run.json`` record the best result and the
  full run configuration.
