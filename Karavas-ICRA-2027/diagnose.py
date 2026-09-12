#!/usr/bin/env python3
"""Why do the networks lose to persistence? Isolate the cause.

Trains the fast CNN backbone on one split under a few configurations and
compares each against the trivial baselines on the same held-out subjects.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import td_pipeline as tp
from td_models import MODEL_SPECS, training_callbacks


def baselines(fold):
    """Persistence and linear extrapolation on this fold's test subjects."""
    out = {}
    for sid, frame in fold.test_frames.items():
        ang = frame[tp.ANGLE_COLS].to_numpy(float)
        gt = ang[tp.WINDOW:]
        pers = ang[tp.WINDOW - 1:-1]
        lin = 2 * ang[tp.WINDOW - 1:-1] - ang[tp.WINDOW - 2:-2]
        out[sid] = (
            float(np.mean(np.abs(pers - gt))),
            float(np.mean(np.abs(lin - gt))),
        )
    return out


def evaluate(model, fold, kind):
    maes = []
    for sid, frame in fold.test_frames.items():
        feats, targets = fold.features(frame, kind)
        x, y = tp.make_windows(feats, targets)
        pred = np.asarray(model.predict(x, verbose=0))
        if pred.ndim == 3:
            pred = pred[:, 0, :]
        pred = fold.scaler_ang.inverse_transform(pred)
        gt = fold.scaler_ang.inverse_transform(y)
        maes.append(float(np.mean(np.abs(pred - gt))))
    return float(np.mean(maes))


def run(label, key, fold, epochs, batch_size, use_callbacks=True):
    spec = MODEL_SPECS[key]
    x_train, y_train = fold.train_xy(spec.kind)
    x_val, y_val = fold.val_xy(spec.kind)

    model = spec.build()
    t0 = time.time()
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=training_callbacks() if use_callbacks else [],
        verbose=0,
    )
    mae = evaluate(model, fold, spec.kind)
    print(f"  {label:<34} train_loss={hist.history['loss'][-1]:.4f} "
          f"val_loss={min(hist.history['val_loss']):.4f} "
          f"epochs={len(hist.history['loss']):>3} "
          f"test_MAE={mae:.2f} deg  ({time.time() - t0:.0f}s)")
    return mae


def main() -> None:
    raw = tp.load_all_subjects()
    split = tp.make_repeated_splits(sorted(raw), n_repeats=1)[0]
    print(f"split: test={split.test} val={split.val}\n")

    key = sys.argv[1] if len(sys.argv) > 1 else "timestamp_cnn"
    print(f"model under test: {MODEL_SPECS[key].label}\n")

    for aug in (0, 2, 7):
        fold = tp.build_fold(split, raw, aug_rounds=aug)
        if aug == 0:
            b = baselines(fold)
            pers = np.mean([v[0] for v in b.values()])
            lin = np.mean([v[1] for v in b.values()])
            print(f"BASELINES on these subjects: persistence {pers:.2f} deg, "
                  f"linear extrapolation {lin:.2f} deg\n")
        n = fold.train_xy(MODEL_SPECS[key].kind)[0].shape[0]
        print(f"aug_rounds={aug}  ({n} training windows)")
        run(f"default (bs=256, early stop)", key, fold, 200, 256)
        run(f"small batch (bs=32)", key, fold, 200, 32)
        print()


if __name__ == "__main__":
    main()
