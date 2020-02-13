import numpy as np
import pandas as pd
import pomegranate as pg
from retrying import retry, RetryError
from tqdm import tqdm
import lib.utils


def generate_traces(
    n_traces,
    state_means="random",
    random_k_states_max=2,
    min_state_diff=0.1,
    D_lifetime=400,
    A_lifetime=200,
    blink_prob=0.05,
    bleed_through=0,
    aa_mismatch=(-0.3, 0.3),
    trace_length=200,
    trans_prob=0.1,
    noise=0.08,
    trans_mat=None,
    au_scaling_factor=1,
    aggregation_prob=0.1,
    max_aggregate_size=100,
    null_fret_value=-1,
    acceptable_noise=0.25,
    scramble_prob=0.3,
    add_gamma_noise=True,
    merge_labels=False,
    discard_unbleached=False,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    n_traces:
        Number of traces to generate
    state_means:
        Mean FRET value. Add multiple values for multiple states
    random_k_states_max:
        If state_means = "random", randomly selects at most k FRET states
    min_state_diff:
        If state_means = "random", randomly spaces states with a minimum
        distance
    D_lifetime:
        Lifetime of donor fluorophore, as drawn from exponential distribution.
        Set to None if fluorophore shouldn't bleach.
    A_lifetime:
        Lifetime of acceptor fluorophore, as drawn from exponential
        distribution. Set to None if fluorophore shouldn't bleach.
    blink_prob:
        Probability of observing photoblinking in a trace.
    bleed_through:
        Donor bleed-through into acceptor channel, as a fraction of the signal.
    aa_mismatch:
        Acceptor-only intensity mis-correspondence, as compared to DD+DA signal.
        Set as a value or range. A value e.g. 0.1 corresponds to 110% of the
        DD+DA signal. A range (-0.3, 0.3) corresponds to 70% to 130% of DD+DA
        signal.
    trace_length:
        Simulated recording length of traces. All traces will adhere to this
        length.
    trans_prob:
        Probability of transitioning from one state to another, given the
        transition probability matrix. This can also be overruled by a supplied
        transition matrix (see trans_mat parameter).
    noise:
        Noise added to a trace, as generated from a Normal(0, sigma)
        distribution. Sigma can be either a value or range.
    trans_mat:
        Transition matrix to be provided instead of the quick trans_prob
        parameter.
    au_scaling_factor:
        Arbitrary unit scaling factor after trace generation. Can be value or
        range.
    aggregation_prob:
        Probability of trace being an aggregate. Note that this locks the
        labelled molecule in a random, fixed FRET state.
    max_aggregate_size:
        Maximum number of labelled molecules in an aggregate.
    null_fret_value:
        Whether to set a specific value for the no-longer-viable *ground truth*
        FRET, e.g. -1, to easily locate it for downstream processing.
    acceptable_noise:
        Maximum acceptable noise level before trace is labelled as "noisy". If
        acceptable_noise is above the upper range of noise, no "noisy" traces
        will be generated.
    scramble_prob:
        Probability that the trace will end up being scrambled. This stacks with
        aggregation.
    add_gamma_noise:
        Multiply centered Gamma(1, 0.11) to each frame's noise, to make the data
        appear less synthetic
    merge_labels:
        Merges (dynamic, static) and (aggregate, noisy, scrambled) to deal with
        binary labels only
    discard_unbleached:
        Whether to discard traces that don't fully bleach to background.
    """

    def _E(DD, DA):
        return DA / (DD + DA)

    def _S(DD, DA, AA):
        return (DD + DA) / (DD + DA + AA)

    def _DD(E):
        return 1 - E

    def _DA(DD, E):
        return -(DD * E) / (E - 1)

    def _AA(E):
        return np.ones(len(E))

    @retry
    def _generate_state_means(min_diff, k_states):
        """Returns random values and retries if they are too closely spaced"""
        states = np.random.uniform(0.01, 0.99, k_states)
        diffs = np.diff(sorted(states))
        if any(diffs < min_diff):
            raise RetryError
        return states

    def _generate_fret_states(kind, state_means, trans_mat, trans_prob):
        """Creates artificial FRET states"""
        if all(isinstance(s, float) for s in state_means):
            kind = "defined"

        rand_k_states = np.random.randint(1, random_k_states_max + 1)

        if kind == "random":
            k_states = rand_k_states
            state_means = _generate_state_means(min_state_diff, k_states)
        elif kind == "aggregate":
            state_means = np.random.uniform(0, 1)
            k_states = 1
        else:
            if np.size(state_means) <= random_k_states_max:
                # Pick the same amount of k states as state means given
                k_states = np.size(state_means)
            else:
                # Pick no more than k_states_max from the state means (e.g.
                # given [0.1, 0.2, 0.3, 0.4, 0.5] use only
                # random_k_states_max of these)
                k_states = rand_k_states
                state_means = np.random.choice(
                    state_means, size=k_states, replace=False
                )

        if type(state_means) == float:
            dists = [pg.NormalDistribution(state_means, 0)]
        else:
            dists = [pg.NormalDistribution(m, 0) for m in state_means]

        starts = np.array([1 / k_states] * k_states)

        lib.utils.random_seed_mp()
        np.random.shuffle(dists)

        # Generate arbitrary transition matrix
        if trans_mat is None:
            trans_mat = np.empty([k_states, k_states])
            trans_mat.fill(trans_prob)
            np.fill_diagonal(trans_mat, 1 - trans_prob)

            # Make sure that each row/column sums to exactly 1
            if trans_prob != 0:
                remaining_prob = 1 - trans_mat.sum(axis=0)
                trans_mat[trans_mat == 0] += remaining_prob

        # Generate HMM model
        model = pg.HiddenMarkovModel.from_matrix(
            trans_mat, distributions=dists, starts=starts
        )
        model.bake()

        E_true = np.array(model.sample(trace_length))
        return E_true

    def _generate_trace(*args):
        """Function to generate a single trace for parallel loop"""
        nonlocal LABELS
        nonlocal scramble_prob

        trans_prob, au_scaling_factor, noise, bleed_through, aa_mismatch, i = [
            np.array(arg) for arg in args
        ]

        name = [i.tolist()] * trace_length
        frames = np.arange(1, trace_length + 1, 1)

        if np.random.uniform(0, 1) < aggregation_prob:
            aggregated = True
            E_true = _generate_fret_states(
                kind="aggregate",
                trans_mat=trans_mat,
                trans_prob=0,
                state_means=state_means,
            )
            if max_aggregate_size >= 2:
                aggregate_size = np.random.randint(2, max_aggregate_size + 1)
            else:
                raise ValueError("Can't have an aggregate of size less than 2")
            np.random.seed()
            n_pairs = np.random.poisson(aggregate_size)
            if n_pairs == 0:
                n_pairs = 2
        else:
            aggregated = False
            n_pairs = 1
            trans_prob = np.random.uniform(trans_prob.min(), trans_prob.max())
            E_true = _generate_fret_states(
                kind=state_means,
                trans_mat=trans_mat,
                trans_prob=trans_prob,
                state_means=state_means,
            )

        DD_total, DA_total, AA_total = [], [], []
        first_bleach_all = []
        for j in range(n_pairs):
            np.random.seed()
            if D_lifetime is not None:
                bleach_D = (
                    np.ceil(np.random.exponential(D_lifetime, 1))
                    .astype(int)
                    .squeeze()
                )
            else:
                bleach_D = trace_length + 1

            if A_lifetime is not None:
                bleach_A = (
                    np.ceil(np.random.exponential(A_lifetime, 1))
                    .astype(int)
                    .squeeze()
                )
            else:
                bleach_A = trace_length + 1

            first_bleach = min(bleach_D, bleach_A)
            first_bleach_all.append(first_bleach)

            # Calculate from underlying E
            DD = _DD(E_true)
            DA = _DA(DD, E_true)
            AA = _AA(E_true)

            # In case AA intensity doesn't correspond exactly to donor
            # experimentally (S will be off)
            AA += np.random.uniform(aa_mismatch.min(), aa_mismatch.max(), 1)

            # Donor bleaches first
            if first_bleach == bleach_D:
                DD[bleach_D:] = 0  # Donor bleaches
                DA[
                    bleach_D:
                ] = 0  # DA goes to zero because no energy is transferred

            # Acceptor bleaches first
            elif first_bleach == bleach_A:
                DD[bleach_A:bleach_D] = 1  # Donor is 1 when there's no acceptor
                if aggregated and n_pairs <= 2:
                    spike_len = np.min((np.random.randint(2, 10), bleach_D))
                    DD[bleach_A : bleach_A + spike_len] = 2

            # No matter what, zero each signal after its own bleaching
            for s, b in zip((DD, DA, AA), (bleach_D, bleach_A, bleach_A)):
                s[b:] = 0

            DD_total.append(DD)
            DA_total.append(DA)
            AA_total.append(AA)

        first_bleach_all = first_bleach_all[0]
        totals = (DD_total, DA_total, AA_total)
        signals = [np.sum(x, axis=0) for x in totals]
        label = np.zeros(trace_length)

        DD, DA, AA = signals

        if not aggregated:
            p = np.random.uniform(0, 1)
            if p < blink_prob:
                if first_bleach_all > 1:
                    blink_start = np.random.randint(1, trace_length)
                    blink_time = np.random.randint(1, 15)
                    if blink_start + blink_time < first_bleach_all:
                        # Set intensities
                        DA[blink_start : blink_start + blink_time] = 0
                        AA[blink_start : blink_start + blink_time] = 0
                        E_true[
                            blink_start : blink_start + blink_time
                        ] = null_fret_value

        signals = DD, DA, AA

        for n in range(3):
            if aggregated:
                label[signals[n] != LABELS["bleached"]] = LABELS["aggregate"]
            else:
                label[signals[n] != LABELS["bleached"]] = LABELS["dynamic"]

        for n in range(3):  # Label any bleached points zero
            label[signals[n] == LABELS["bleached"]] = LABELS["bleached"]

        if label.all() != LABELS["bleached"]:
            first_bleach_all = None
        else:
            first_bleach_all = label.argmin()

        if not aggregated:
            if first_bleach_all is not None:
                E_true[first_bleach_all:] = null_fret_value
        else:
            E_true = [null_fret_value] * trace_length

        p = np.random.uniform(0, 1)
        if p < scramble_prob:  # do weird stuff to the trace
            if n_pairs <= 2:
                # Modify trace with weird correlations
                modify_trace = np.random.choice(("DD", "DA", "AA"))
                if modify_trace == "DD":
                    DD[DD != 0] = 1
                    sinwave = np.sin(
                        np.linspace(-10, np.random.randint(0, 1), len(DD))
                    )  # Create a sign wave and merge with trace
                    sinwave[DD == 0] = 0
                    sinwave = sinwave ** np.random.randint(5, 10)
                    DD += sinwave * 0.4
                    DD[DD < 0] = 0  # first signal
                elif modify_trace == "DA":
                    DA *= AA * np.random.uniform(0.7, 1, 1)
                elif modify_trace == "AA":
                    # Correlate heavily
                    AA *= DA * np.random.uniform(0.7, 1, 1)
                else:
                    pass

                # Add dark state
                add_dark = np.random.choice(("add", "noadd"))
                if add_dark == "add":
                    dark_state_start = np.random.randint(0, 50)
                    dark_state_time = np.random.randint(10, 50)
                    dark_state_end = dark_state_start + dark_state_time
                    DD[dark_state_start:dark_state_end] = 0
                else:
                    pass

                # Add noise
                p = np.random.uniform(0, 1)
                if p < 0.1:
                    noise_start = np.random.randint(1, trace_length)
                    noise_time = np.random.randint(10, 50)
                    noise_end = noise_start + noise_time
                    if noise_end > trace_length:
                        noise_end = trace_length

                    DD[noise_start:noise_end] *= np.random.normal(
                        1, 1, noise_end - noise_start
                    )

                # Flip traces
                flip_trace = np.random.choice(("flipDD", "flipDA", "flipAA"))
                if flip_trace == "flipDD":
                    DD = np.flip(DD)
                elif flip_trace == "flipAA":
                    AA = np.flip(AA)
                elif flip_trace == "flipDA":
                    DA = np.flip(DA)
                else:
                    pass

                DD[DD < 0] = 0  # fix multiplication below zero
                DA[DA < 0] = 0
                AA[AA < 0] = 0

                label[
                    (label == LABELS["dynamic"]) | (label == LABELS["static"])
                ] = LABELS["scramble"]

                # Figure out bleached places:
                label[DD == 0] = LABELS["bleached"]
                label[AA == 0] = LABELS["bleached"]

        DD_bleed = np.random.uniform(bleed_through.min(), bleed_through.max())
        DA[DD != 0] += DD_bleed

        signals = DD, DA, AA

        noise = np.random.uniform(noise.min(), noise.max())
        observed_states = np.unique(E_true[E_true != null_fret_value])

        signals = [s + np.random.normal(0, noise, len(s)) for s in signals]

        if add_gamma_noise:
            for s in signals:
                gnoise = np.random.gamma(1, noise * 1.1, len(s))
                s += gnoise
                s -= np.mean(gnoise)

        au_scaling_factor = np.random.uniform(
            au_scaling_factor.min(), au_scaling_factor.max()
        )
        signals = [s * au_scaling_factor for s in signals]

        DD, DA, AA = signals

        E_obs = _E(DD, DA)
        S_obs = _S(DD, DA, AA)

        E_nb = E_obs[E_true != null_fret_value]

        if len(observed_states) == 1:
            if np.std(E_nb) > acceptable_noise:
                label[label == LABELS["dynamic"]] = LABELS["noisy"]
            else:
                # else, keep as single-state traces
                label[label == LABELS["dynamic"]] = LABELS["static"]
        else:
            for s in observed_states:
                if np.std(E_obs[E_true == s]) > acceptable_noise:
                    label[
                        (label == LABELS["dynamic"])
                        | (label == LABELS["static"])
                    ] = LABELS["noisy"]

        if merge_labels:
            label[
                (label != LABELS["dynamic"]) | (label != LABELS["static"])
            ] = 0
            label[
                (label == LABELS["dynamic"]) | (label == LABELS["static"])
            ] = 1

        if discard_unbleached:
            if label[-1] != LABELS["bleached"]:
                return pd.DataFrame()

        s = pd.DataFrame(
            {
                "DD": DD,
                "DA": DA,
                "AA": AA,
                "E": E_obs,
                "E_true": E_true,
                "S": S_obs,
                "frame": frames,
                "name": name,
                "fb": [first_bleach_all] * trace_length,
                "label": label,
            }
        )
        s.replace([np.inf, -np.inf], np.nan, inplace=True)
        s.fillna(method="pad", inplace=True)
        return s

    LABELS = {
        "bleached": 0,
        "aggregate": 1,
        "dynamic": 2,
        "static": 3,
        "noisy": 4,
        "scramble": 5,
    }

    processes = tqdm(range(n_traces))

    traces = [
        _generate_trace(
            trans_prob, au_scaling_factor, noise, bleed_through, aa_mismatch, i
        )
        for i in processes
    ]

    if len(traces) > 1:
        traces = pd.concat(traces)
    else:
        traces = traces[0]

    return traces
