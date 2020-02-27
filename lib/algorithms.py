import numpy as np
import pandas as pd
import pomegranate as pg
from retrying import RetryError, retry
import parmap
import lib.utils

from lib.utils import global_function


def generate_traces(
    n_traces,
    state_means = "random",
    random_k_states_max = 2,
    min_state_diff = 0.1,
    D_lifetime = 400,
    A_lifetime = 200,
    blink_prob = 0.05,
    bleed_through = 0,
    aa_mismatch = (-0.3, 0.3),
    trace_length = 200,
    trans_prob = 0.1,
    noise = 0.08,
    trans_mat = None,
    au_scaling_factor = 1,
    aggregation_prob = 0.1,
    max_aggregate_size = 100,
    null_fret_value = -1,
    acceptable_noise = 0.25,
    scramble_prob = 0.3,
    gamma_noise_prob = 0.5,
    merge_labels = False,
    discard_unbleached = False,
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
    gamma_noise_prob:
        Probability to multiply centered Gamma(1, 0.11) to each frame's noise,
        to make the data appear less synthetic
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
    def generate_state_means(min_acceptable_diff, k_states):
        """Returns random values and retries if they are too closely spaced"""
        states = np.random.uniform(0.01, 0.99, k_states)
        diffs = np.diff(sorted(states))
        if any(diffs < min_acceptable_diff):
            raise RetryError
        return states

    def generate_fret_states(kind, state_means, trans_mat, trans_prob):
        """Creates artificial FRET states"""
        if all(isinstance(s, float) for s in state_means):
            kind = "defined"

        rand_k_states = np.random.randint(1, random_k_states_max + 1)

        if kind == "random":
            k_states = rand_k_states
            state_means = generate_state_means(min_state_diff, k_states)
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
                    state_means, size = k_states, replace = False
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
                stay_prob = 1 - trans_prob
                remaining_prob = 1 - trans_mat.sum(axis = 0)
                trans_mat[trans_mat == stay_prob] += remaining_prob

        # Generate HMM model
        model = pg.HiddenMarkovModel.from_matrix(
            trans_mat, distributions = dists, starts = starts
        )
        model.bake()

        E_true = np.array(model.sample(trace_length))
        return E_true

    def scramble(DD, DA, AA, cls, label):
        """Scramble trace for model robustness"""

        modify_trace = np.random.choice(("DD", "DA", "AA"))
        if modify_trace == "DD":
            c = DD
        elif modify_trace == "DA":
            c = DA
        elif modify_trace == "AA":
            c = AA
        else:
            raise ValueError

        c[c != 0] = 1
        # Create a sign wave and merge with trace
        sinwave = np.sin(np.linspace(-10, np.random.randint(0, 1), len(DD)))
        sinwave[c == 0] = 0
        sinwave = sinwave ** np.random.randint(5, 10)
        c += sinwave * 0.4
        # Fix negatives
        c = np.abs(c)

        # Correlate heavily
        DA *= AA * np.random.uniform(0.7, 1)
        AA *= DA * np.random.uniform(0.7, 1)
        DD *= AA * np.random.uniform(0.7, 1)

        # Add dark state
        add_dark = np.random.choice(("add", "noadd"))
        if add_dark == "add":
            dark_state_start = np.random.randint(0, 40)
            dark_state_time = np.random.randint(10, 40)
            dark_state_end = dark_state_start + dark_state_time
            DD[dark_state_start:dark_state_end] = 0

        # Add noise
        if np.random.uniform(0, 1) < 0.1:
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

        DD, DA, AA = [np.abs(x) for x in (DD, DA, AA)]

        label.fill(cls["scramble"])
        return DD, DA, AA, label

    @global_function
    def generate_single_trace(*args):
        """Function to generate a single trace"""
        (
            i,
            trans_prob,
            au_scaling_factor,
            noise,
            bleed_through,
            aa_mismatch,
            scramble_prob,
        ) = [np.array(arg) for arg in args]

        # Simple table to keep track of labels
        cls = {
            "bleached" : 0,
            "aggregate": 1,
            "noisy"    : 2,
            "scramble" : 3,
            "1-state"  : 4,
            "2-state"  : 5,
            "3-state"  : 6,
            "4-state"  : 7,
            "5-state"  : 8,
        }

        name = [i.tolist()] * trace_length
        frames = np.arange(1, trace_length + 1, 1)

        if np.random.uniform(0, 1) < aggregation_prob:
            is_aggregated = True
            E_true = generate_fret_states(
                kind = "aggregate",
                trans_mat = trans_mat,
                trans_prob = 0,
                state_means = state_means,
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
            is_aggregated = False
            n_pairs = 1
            trans_prob = np.random.uniform(trans_prob.min(), trans_prob.max())
            E_true = generate_fret_states(
                kind = state_means,
                trans_mat = trans_mat,
                trans_prob = trans_prob,
                state_means = state_means,
            )

        DD_total, DA_total, AA_total = [], [], []
        first_bleach_all = []

        for j in range(n_pairs):
            np.random.seed()
            if D_lifetime is not None:
                bleach_D = int(np.ceil(np.random.exponential(D_lifetime)))
            else:
                bleach_D = None

            if A_lifetime is not None:
                bleach_A = int(np.ceil(np.random.exponential(A_lifetime)))
            else:
                bleach_A = None

            first_bleach = lib.utils.min_none((bleach_D, bleach_A))
            first_bleach_all.append(first_bleach)

            # Calculate from underlying E
            DD = _DD(E_true)
            DA = _DA(DD, E_true)
            AA = _AA(E_true)

            # In case AA intensity doesn't correspond exactly to donor
            # experimentally (S will be off)
            AA += np.random.uniform(aa_mismatch.min(), aa_mismatch.max())

            # If donor bleaches first
            if first_bleach is not None:
                if first_bleach == bleach_D:
                    # Donor bleaches
                    DD[bleach_D:] = 0
                    # DA goes to zero because no energy is transferred
                    DA[bleach_D:] = 0

                # If acceptor bleaches first
                elif first_bleach == bleach_A:
                    # Donor is 1 when there's no acceptor
                    DD[bleach_A:bleach_D] = 1
                    if is_aggregated and n_pairs <= 2:
                        # Sudden spike for small aggregates to mimic
                        # observations
                        spike_len = np.min((np.random.randint(2, 10), bleach_D))
                        DD[bleach_A: bleach_A + spike_len] = 2

            # No matter what, zero each signal after its own bleaching
            if bleach_D is not None:
                DD[bleach_D:] = 0
            if bleach_A is not None:
                DA[bleach_A:] = 0
                AA[bleach_A:] = 0

            # Append to total fluorophore intensity per channel
            DD_total.append(DD)
            DA_total.append(DA)
            AA_total.append(AA)

        DD, DA, AA = [np.sum(x, axis = 0) for x in
                      (DD_total, DA_total, AA_total)]

        # Initialize -1 label for whole trace
        label = np.zeros(trace_length)
        label.fill(-1)

        # Calculate when a channel is bleached. For aggregates, it's when a
        # fluorophore channel hits 0 from bleaching (because 100% FRET not
        # considered possible)
        if is_aggregated:
            # First bleaching for
            bleach_DD_all = np.argmax(DD == 0)
            bleach_DA_all = np.argmax(DA == 0)
            bleach_AA_all = np.argmax(AA == 0)

            # Find first bleaching overall
            first_bleach_all = lib.utils.min_none(
                (bleach_DD_all, bleach_DA_all, bleach_AA_all)
            )
            if first_bleach_all == 0:
                first_bleach_all = None
            label.fill(cls["aggregate"])
        else:
            # Else simply check whether DD or DA bleaches first from lifetimes
            first_bleach_all = lib.utils.min_none(first_bleach_all)

        # Save unblinked fluorophores to calculate E_true
        DD_no_blink, DA_no_blink = DD.copy(), DA.copy()

        # No blinking in aggregates (excessive/complicated)
        if not is_aggregated:
            if np.random.uniform(0, 1) < blink_prob:
                blink_start = np.random.randint(1, trace_length)
                blink_time = np.random.randint(1, 15)

                # Blink either donor or acceptor
                if np.random.uniform(0, 1) < 0.5:
                    DD[blink_start: (blink_start + blink_time)] = 0
                    DA[blink_start: (blink_start + blink_time)] = 0
                else:
                    DA[blink_start: (blink_start + blink_time)] = 0
                    AA[blink_start: (blink_start + blink_time)] = 0

        if first_bleach_all is not None:
            label[first_bleach_all:] = cls["bleached"]
            E_true[first_bleach_all:] = null_fret_value

        for x in (DD, DA, AA):
            # Bleached points get label 0
            label[x == 0] = cls["bleached"]

        if is_aggregated:
            first_bleach_all = np.argmin(label)
            if first_bleach_all == 0:
                first_bleach_all = None

        # Scramble trace, but only if contains 1 or 2 pairs (diminishing
        # effect otherwise)
        is_scrambled = False
        if np.random.uniform(0, 1) < scramble_prob and n_pairs <= 2:
            DD, DA, AA, label = scramble(
                DD = DD, DA = DA, AA = AA, cls = cls, label = label
            )
            is_scrambled = True

        # Figure out bleached places before true signal is modified:
        is_bleached = np.zeros(trace_length)
        for x in (DD, DA, AA):
            is_bleached[x == 0] = 1

        # Add donor bleed-through
        DD_bleed = np.random.uniform(bleed_through.min(), bleed_through.max())
        DA[DD != 0] += DD_bleed

        # Re-adjust E_true to match offset caused by correction factors
        # so technically it's not the true, corrected FRET, but actually the
        # un-noised
        E_true[E_true != null_fret_value] = _E(
            DD_no_blink[E_true != null_fret_value],
            DA_no_blink[E_true != null_fret_value],
        )

        # Add gaussian noise
        noise = np.random.uniform(noise.min(), noise.max())
        x = [s + np.random.normal(0, noise, len(s)) for s in (DD, DA, AA)]

        # Add centered gamma noise
        if np.random.uniform(0, 1) < gamma_noise_prob:
            for signal in x:
                gnoise = np.random.gamma(1, noise * 1.1, len(signal))
                signal += gnoise
                signal -= np.mean(gnoise)

        # Scale trace to AU units and calculate observed E and S as one would
        # in real experiments
        au_scaling_factor = np.random.uniform(
            au_scaling_factor.min(), au_scaling_factor.max()
        )
        DD, DA, AA = [s * au_scaling_factor for s in x]

        E_obs = _E(DD, DA)
        S_obs = _S(DD, DA, AA)

        # FRET from fluorophores that aren't bleached
        E_unbleached = E_obs[:first_bleach_all]
        E_unbleached_true = E_true[:first_bleach_all]

        # Count actually observed states, because a slow system might not
        # transition in the observation window
        observed_states = np.unique(E_true[E_true != null_fret_value])

        # Calculate noise level for each FRET state, and check if it
        # surpasses the limit
        is_noisy = False
        for state in observed_states:
            noise_level = np.std(E_unbleached[E_unbleached_true == state])
            if noise_level > acceptable_noise:
                label[label != cls["bleached"]] = cls["noisy"]
                is_noisy = True

        # For all FRET traces, assign the number of states observed
        if not any((is_noisy, is_aggregated, is_scrambled)):
            for i in range(5):
                k_states = i + 1
                if len(observed_states) == k_states:
                    label[label != cls["bleached"]] = cls[
                        "{}-state".format(k_states)
                    ]

        # Bad traces don't contain FRET
        if any((is_noisy, is_aggregated, is_scrambled)):
            E_true.fill(-1)

        # Everything that isn't FRET is 0, and FRET is 1
        if merge_labels:
            label[label <= 3] = 0
            label[label >= 4] = 1

        if discard_unbleached:
            if label[-1] != cls["bleached"]:
                return pd.DataFrame()

        if label[0] in [4, 5, 6, 7, 8]:
            min_diff = np.min(np.diff(np.unique(E_unbleached_true)))
        else:
            min_diff = np.nan

        # Columns pre-fixed with underscore contain metadata, and only the
        # first value should be used (repeated because table structure)
        trace = pd.DataFrame(
            {
                "DD"            : DD,
                "DA"            : DA,
                "AA"            : AA,
                "E"             : E_obs,
                "E_true"        : E_true,
                "S"             : S_obs,
                "frame"         : frames,
                "name"          : name,
                "label"         : label,
                "_bleaches_at"   : first_bleach_all.repeat(trace_length),
                "_noise_level"   : noise.repeat(trace_length),
                "_min_state_diff": min_diff.repeat(trace_length),
            }
        )
        trace.replace([np.inf, -np.inf], np.nan, inplace = True)
        trace.fillna(method = "pad", inplace = True)
        return trace

    traces = parmap.map(
        generate_single_trace,
        range(n_traces),
        trans_prob,
        au_scaling_factor,
        noise,
        bleed_through,
        aa_mismatch,
        scramble_prob,
    )

    if len(traces) > 1:
        traces = pd.concat(traces)
    else:
        traces = traces[0]

    return traces
