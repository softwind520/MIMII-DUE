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
