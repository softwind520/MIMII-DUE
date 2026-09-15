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

## Train the section/domain-conditional diffusion model

``diffusion.yaml`` remains the reproducible unconditional experiment used by
the existing fan checkpoint. The new ``conditional.yaml`` enables real section
and domain conditioning and writes to separate ``checkpoints_conditional/`` and
``outputs_conditional/`` directories, so it cannot overwrite the unconditional
baseline.

The condition projector concatenates learned section and domain embeddings,
projects them to the timestep-embedding dimension, and adds the result to the
timestep embedding used for scale-shift modulation in every U-Net residual
block. During training, both labels are jointly replaced by learned unknown
labels with probability 0.1. This classifier-free condition dropout prevents
the denoiser from depending completely on metadata and provides a defined
fallback for unavailable labels.

Validate the conditional data contract and run one real optimization step:

```bash
python 00_train.py --config conditional.yaml --check-data --machine-type fan --max-files 12
CUDA_VISIBLE_DEVICES=1 python 00_train.py --config conditional.yaml --smoke-test --machine-type fan
```

The smoke checkpoint is isolated under
``checkpoints_conditional/smoke/fan/``. Start the full fan training with:

```bash
CUDA_VISIBLE_DEVICES=1 python 00_train.py --config conditional.yaml --machine-type fan
```

Use ``--resume`` only to continue an interrupted full conditional run. Once the
100 epochs finish, evaluate it with the same frozen two-component GMM protocol
used by the unconditional model:

```bash
CUDA_VISIBLE_DEVICES=1 python 01_test.py --config conditional.yaml --fan-gmm --machine-type fan
```

This produces the controlled comparison:

```text
checkpoints/fan/ema.pt              unconditional model
checkpoints_conditional/fan/ema.pt  section/domain-conditional model
outputs/fan_gmm/                    unconditional GMM result
outputs_conditional/fan_gmm/        conditional GMM result
```

The Log-Mel extraction settings are unchanged, so both experiments safely share
the existing ``feature_cache/``.

## Train the section-only conditional diffusion model

``section_only.yaml`` is the controlled ablation prompted by the weak target-domain
result of explicit domain conditioning. It keeps learned section conditioning and
domain-balanced source/target sampling, but the domain label is not passed to the
U-Net. Its checkpoints and results are isolated under
``checkpoints_section_only/`` and ``outputs_section_only/``.

Validate the data pipeline and perform the disposable one-step training check:

```bash
python 00_train.py --config section_only.yaml --check-data --machine-type fan --max-files 12
CUDA_VISIBLE_DEVICES=1 python 00_train.py --config section_only.yaml --smoke-test --machine-type fan
```

After the smoke checkpoint succeeds, it may be deleted because it is not a
scientific result. Keep the automated tests and the ``--smoke-test`` option for
future model changes. Start the full training from a clean section-only directory:

```bash
CUDA_VISIBLE_DEVICES=1 python 00_train.py --config section_only.yaml --machine-type fan
```

Evaluate the trained model with exactly the same frozen GMM protocol:

```bash
CUDA_VISIBLE_DEVICES=1 python 01_test.py --config section_only.yaml --fan-gmm --machine-type fan \
  --gmm-start-step 400 --gmm-patch-hop 32 --gmm-residual-modes signed \
  --gmm-scopes section --gmm-covariances diag --gmm-components 2
```

The three controlled experiments are therefore:

```text
diffusion.yaml      no metadata condition
conditional.yaml    section + domain conditions
section_only.yaml   section condition only; domain-balanced sampling retained
```

## Expand the section-only model to all machines

The training engine already creates one independent denoiser per machine. Keep
the completed fan checkpoint and train the remaining four machines sequentially:

```bash
python 00_train.py --config section_only.yaml --check-data --max-files 12
CUDA_VISIBLE_DEVICES=1 python 00_train.py --config section_only.yaml \
  --machine-type gearbox --machine-type pump --machine-type slider \
  --machine-type valve
```

Each machine writes to its own directory under ``checkpoints_section_only/``.
The command does not touch ``checkpoints_section_only/fan/`` because fan is not
included in the machine list. If a run is interrupted, resume only that machine
with ``--machine-type MACHINE --resume`` before starting the remaining machines.

After all five ``ema.pt`` files exist, run the frozen GMM protocol on every
machine. The new ``--gmm`` mode defaults to the fan-selected signed residual,
section-specific, diagonal two-component GMM, so it does not perform a new
hyperparameter search:

```bash
CUDA_VISIBLE_DEVICES=1 python 01_test.py --config section_only.yaml --gmm
```

Individual outputs are written to ``outputs_section_only/MACHINE_gmm/``. The
cross-machine report is written directly under ``outputs_section_only/``:

- ``gmm_all_summary.csv`` and ``gmm_all_best.json`` contain the harmonic means
  over every machine/section/domain AUC and pAUC;
- ``gmm_all_groups.csv`` contains all per-group metrics with the machine name;
- ``gmm_all_run.json`` records all checkpoints and evaluation settings.

``--fan-gmm`` remains available for reproducing the original fan-only sweep,
while ``--gmm --machine-type MACHINE`` can evaluate a selected machine.
