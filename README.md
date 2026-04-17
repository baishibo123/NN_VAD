## What changed relative to the starter version

1. **Official LibriSpeech splits**
   - Training: `train-clean-100`
   - Validation: `dev-clean`
   - Test: `test-clean`
   - `prepare_manifest.py` prints speaker-overlap counts so you can explicitly show they are zero.

2. **Fixed MUSAN custom split**
   - MUSAN does not provide official train/val/test splits.
   - This code uses a fixed split by category:
     - train -> `music`
     - val -> `noise`
     - test -> `speech`

3. **Ground-truth label generation is explicit**
   - Labels are not guessed.
   - For each synthetic clip, speech is inserted at a known sample offset.
   - Frame-level labels are generated from the exact speech insertion boundaries.

4. **Balanced speech / non-speech examples**
   - The dataset now generates both:
     - speech+noise clips
     - pure non-speech clips
   - This avoids the earlier issue where TN could collapse to zero.

5. **Multiple SNR evaluation**
   - The evaluation script reports results separately for:
     - noise-only
     - 20 dB
     - 10 dB
     - 0 dB
     - all conditions combined

6. **Energy-threshold tuning on validation data**
   - The energy baseline threshold is tuned on the validation set first, then applied to the test set.

## Recommended commands

### 1. Build manifests

```bash
python prepare_manifest.py \
  --librispeech_root /Users/caitlyn/Desktop/CS6140/LibriSpeech \
  --musan_root /Users/caitlyn/Desktop/CS6140/Musan/musan \
  --out_dir data/manifests
```

### 2. Train

```bash
python train_vad.py \
  --train_manifest data/manifests/train_manifest.jsonl \
  --val_manifest data/manifests/val_manifest.jsonl \
  --epochs 2 \
  --batch_size 8 \
  --save_path outputs/vad_model.pt
```

### 3. Evaluate

```bash
python evaluate_vad.py \
  --val_manifest data/manifests/val_manifest.jsonl \
  --test_manifest data/manifests/test_manifest.jsonl \
  --model_path outputs/vad_model.pt
```
