from pathlib import Path
import numpy as np
import lib.algorithms
import lib.ml
import lib.utils
from time import time


def generate_data(n_traces, n_timesteps=200):
    print("Generating traces...")
    df = lib.algorithms.generate_traces(
        n_traces=n_traces,
        aa_mismatch=(-0.30, 0.30),
        state_means="random",
        random_k_states_max=2,
        min_state_diff=0.2,
        aggregation_prob=0.1,
        max_aggregate_size=7,
        trace_length=n_timesteps,
        trans_prob=(0.00, 0.20),
        blink_prob=0.2,
        bleed_through =(0, 0.15),
        noise=(0.01, 0.30),
        D_lifetime=300,
        A_lifetime=300,
        au_scaling_factor=(10, 10e3),
        null_fret_value=-1,
        acceptable_noise=0.25,
        scramble_prob=0.20,
        add_gamma_noise=True,
        verbose=True,
        discard_unbleached=False,
    )

    X = df[["DD", "DA", "AA"]].values
    y = df["label"].values
    return X, y


def process_and_save(
    X, y, labels_to_binary, balance_classes, outdir, n_timesteps=200
):
    X, y = lib.ml.preprocess_2d_timeseries_seq2seq(
        X=X, y=y, n_timesteps=n_timesteps
    )
    print("Before balance: ", set(y.ravel()))
    ext = False

    if labels_to_binary:
        y = lib.ml.labels_to_binary(y, one_hot=False, to_ones=(2, 3))
        ext = "_binary"
        print("After binarize ", set(y.ravel()))

    if balance_classes:
        X, y = lib.ml.balance_classes(
            X, y, exclude_label_from_limiting=0, frame=0
        )
        print("After balance: ", set(y.ravel()))

    assert not np.any(np.isnan(X))

    for obj, name in zip((X, y), ("X_sim", "y_sim")):
        if ext:
            name += ext
        path = str(Path(outdir).joinpath(name))
        np.savez_compressed(path, obj)

    print(X.shape)
    print("Generated {} traces".format(X.shape[0]))


if __name__ == "__main__":
    DATADIR = "./data"
    N_TRACES = int(1000)
    LABELS_TO_BINARY = False

    start = time()
    X, y = generate_data(n_traces=N_TRACES)
    process_and_save(
        X,
        y,
        labels_to_binary=LABELS_TO_BINARY,
        balance_classes=True,
        outdir=DATADIR,
    )
    end = time()

    print("Time elapsed: {:.1f} s".format(end - start))
