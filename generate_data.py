from pathlib import Path

import numpy as np

import lib.algorithms
import lib.ml
import lib.utils
import lib.plotting
import matplotlib.pyplot as plt
from time import time
from lib.ml import _merge_hmm_labels


def main(n_traces, n_timesteps, merge_hmm_labels, labels_to_binary, balance_classes, outdir, reduce_memory):
    """

    Parameters
    ----------
    n_traces:
        Number of traces to generate
    n_timesteps:
        Length of each trace
    merge_hmm_labels:
        Whether to merge all HMM states above 2 into "dynamic", as the HMM
        predictions don't work so well yet
    labels_to_binary:
        Whether to convert all labels to smFRET/not-smFRET (for each frame)
    balance_classes:
        Whether to balance classes based on the distribution of frame 1 (as
        this changes over time due to bleaching)
    outdir:
        Output directory
    """
    print("Generating traces...")
    start = time()
    X = lib.algorithms.generate_traces(
        n_traces=int(n_traces),
        aa_mismatch=(-0.35, 0.35),
        state_means="random",
        random_k_states_max=4,
        min_state_diff=0.1,
        aggregation_prob=0.15,
        scramble_prob=0.15,
        max_aggregate_size=20,
        trace_length=n_timesteps,
        trans_prob=(0.0, 0.20),
        blink_prob=0.2,
        bleed_through=(0, 0.15),
        noise=(0.01, 0.30),
        acceptable_noise=0.25,
        D_lifetime=500,
        A_lifetime=500,
        scramble_decouple_prob = 0.9,
        falloff_lifetime=500,
        falloff_prob=0.1,
        gamma_noise_prob=0.8,
        au_scaling_factor=(1),
        null_fret_value=-1,
        discard_unbleached=False,
        run_headless_parallel = True,
        return_matrix = False,
        reduce_memory = True,
    )
    stop = time()
    print("spent {:.2f} s to generate".format((stop - start)))

    labels = X["label"].values

    if reduce_memory:
        X = X[["D-Dexc-rw", "A-Dexc-rw", "A-Aexc-rw"]].values
    else:
        X = X[["D-Dexc-rw", "A-Dexc-rw", "A-Aexc-rw", "E", "E_true"]].values

    if np.any(X == -1):
        print(
            "Dataset contains negative E_true. Be careful if using this "
            "for regression!"
        )

    if merge_hmm_labels:
        labels = _merge_hmm_labels(labels)

    X, labels = lib.ml.preprocess_2d_timeseries_seq2seq(
        X=X, y=labels, n_timesteps=n_timesteps
    )
    print("Before balance: ", set(labels.ravel()))
    ext = False

    if labels_to_binary:
        labels = lib.ml.labels_to_binary(labels, one_hot=False, to_ones=(4, 5, 6, 7, 8))
        ext = "_binary"
        print("After binarize ", set(labels.ravel()))

    if balance_classes:
        X, labels = lib.ml.balance_classes(
            X, labels, exclude_label_from_limiting=0, frame=0
        )
        print("After balance:  ", set(labels.ravel()))

    lib.plotting.plot_trace_label_distribution(X=X, y=labels)

    if np.any(np.isnan(X)):
        raise ValueError

    for obj, name in zip((X, labels), ("X_sim", "y_sim")):
        if ext:
            name += ext
        path = str(Path(outdir).joinpath(name))
        np.savez_compressed(path, obj)

    print(X.shape)
    print("Generated {} traces".format(X.shape[0]))

    plt.show()


if __name__ == "__main__":
    main(
        n_traces=int(input("Initial number of traces to generate (will be balanced): ")),
        n_timesteps=300,
        merge_hmm_labels = True,
        balance_classes=True,
        labels_to_binary=False,
        reduce_memory = True,
        outdir="./data",
    )