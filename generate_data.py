from pathlib import Path

import numpy as np

import lib.algorithms
import lib.ml
import lib.utils


def main(n_traces, n_timesteps, labels_to_binary, balance_classes, outdir):
    """

    Parameters
    ----------
    n_traces:
        Number of traces to generate
    n_timesteps:
        Length of each trace
    labels_to_binary:
        Whether to convert all labels to smFRET/not-smFRET (for each frame)
    balance_classes:
        Whether to balance classes based on the distribution of frame 1 (as
        this changes over time due to bleaching)
    outdir:
        Output directory
    """
    n_traces = int(n_traces)

    print("Generating traces...")
    df = lib.algorithms.generate_traces(
        n_traces=n_traces,
        aa_mismatch=(-0.35, 0.35),
        state_means="random",
        random_k_states_max=5,
        min_state_diff=0.1,
        aggregation_prob=0.1,
        max_aggregate_size=7,
        trace_length=n_timesteps,
        trans_prob=(0.00, 0.20),
        blink_prob=0.2,
        bleed_through=(0, 0.15),
        noise=(0.01, 0.30),
        D_lifetime=300,
        A_lifetime=300,
        au_scaling_factor=(10, 10e3),
        null_fret_value=-1,
        acceptable_noise=0.25,
        scramble_prob=0.20,
        add_gamma_noise=True,
        discard_unbleached=False,
    )

    X = df[["DD", "DA", "AA", "E_true"]].values
    labels = df["label"].values

    X, labels = lib.ml.preprocess_2d_timeseries_seq2seq(
        X=X, y=labels, n_timesteps=n_timesteps
    )
    print("Before balance: ", set(labels.ravel()))
    ext = False

    if labels_to_binary:
        labels = lib.ml.labels_to_binary(labels, one_hot=False, to_ones=(2, 3))
        ext = "_binary"
        print("After binarize ", set(labels.ravel()))

    if balance_classes:
        X, labels = lib.ml.balance_classes(
            X, labels, exclude_label_from_limiting=0, frame=0
        )
        print("After balance: ", set(labels.ravel()))

    assert not np.any(np.isnan(X))

    for obj, name in zip((X, labels), ("X_sim", "y_sim")):
        if ext:
            name += ext
        path = str(Path(outdir).joinpath(name))
        np.savez_compressed(path, obj)

    print(X.shape)
    print("Generated {} traces".format(X.shape[0]))


if __name__ == "__main__":
    main(
        n_traces=10000,
        n_timesteps=300,
        balance_classes=True,
        labels_to_binary=False,
        outdir="./data",
    )
