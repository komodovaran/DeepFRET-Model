import os
import matplotlib.ticker
import numpy as np
import pandas as pd
import mlxtend.evaluate
from matplotlib import pyplot as plt
import lib.ml
import lib.utils


def _swap_y_labels(*y):
    """
    Swaps the labels accordingly:

    0: bleached        0: bleached
    1: aggregate       1: aggregate
    2: dynamic         4: noisy
    3: static          5: scrambled
    4: noisy           3: static
    5: scrambled       2: dynamic

    Inputs must be argmaxed and unraveled
    """
    ys = y
    ys = [lib.utils.swap_integers(y, 5, 2) for y in ys]
    ys = [lib.utils.swap_integers(y, 3, 4) for y in ys]
    ys = [lib.utils.swap_integers(y, 4, 5) for y in ys]
    return ys


def plot_losses(logpath, outdir, name, show_only=False):
    """Plots training and validation loss"""
    stats = pd.read_csv(os.path.join(logpath, name + "_training.log")).values
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    axes = axes.ravel()

    epoch = stats[:, 0]
    train_acc = stats[:, 1]
    train_loss = stats[:, 2]
    val_acc = stats[:, 3]
    val_loss = stats[:, 4]

    best_loss = np.argmin(val_loss)

    axes[0].plot(epoch, train_loss, "o-", label="train loss", color="salmon")
    axes[0].plot(epoch, val_loss, "o-", label="val loss", color="lightblue")
    axes[0].axvline(best_loss, color="black", ls="--", alpha=0.5)

    axes[1].plot(
        epoch,
        train_acc,
        "o-",
        label="train acc (best: {:.4f})".format(train_acc.max()),
        color="salmon",
    )
    axes[1].plot(
        epoch,
        val_acc,
        "o-",
        label="val acc (best: {:.4f})".format(val_acc.max()),
        color="lightblue",
    )
    axes[1].axvline(best_loss, color="black", ls="--", alpha=0.5)

    axes[0].legend(loc="lower left")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    if show_only:
        plt.show()
    else:
        plt.savefig(os.path.join(outdir, name + "_loss.pdf"))
        plt.close()


def _plot_confusion_matrix_mlxtend(
    conf_mat,
    hide_spines=False,
    hide_ticks=False,
    figsize=None,
    cmap=None,
    colorbar=False,
    show_absolute=True,
    show_normed=False,
):
    """
    A modified version of mlxtend.plotting.plot_confusion_matrix
    -----------

    Plot a confusion matrix via matplotlib.
    Parameters
    -----------
    conf_mat : array-like, shape = [n_classes, n_classes]
        Confusion matrix from evaluate.confusion matrix.
    hide_spines : bool (default: False)
        Hides axis spines if True.
    hide_ticks : bool (default: False)
        Hides axis ticks if True
    figsize : tuple (default: (2.5, 2.5))
        Height and width of the figure
    cmap : matplotlib colormap (default: `None`)
        Uses matplotlib.pyplot.cm.Blues if `None`
    colorbar : bool (default: False)
        Shows a colorbar if True
    show_absolute : bool (default: True)
        Shows absolute confusion matrix coefficients if True.
        At least one of  `show_absolute` or `show_normed`
        must be True.
    show_normed : bool (default: False)
        Shows normed confusion matrix coefficients if True.
        The normed confusion matrix coefficients give the
        proportion of training examples per class that are
        assigned the correct label.
        At least one of  `show_absolute` or `show_normed`
        must be True.
    Returns
    -----------
    fig, ax : matplotlib.pyplot subplot objects
        Figure and axis elements of the subplot.
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/
    """
    if not (show_absolute or show_normed):
        raise AssertionError("Both show_absolute and show_normed are False")

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype("float") / total_samples

    if figsize is None:
        figsize = (len(conf_mat) * 1.25, len(conf_mat) * 1.25)

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    matshow = ax.matshow(normed_conf_mat, cmap=cmap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                n = matplotlib.ticker.EngFormatter(places=1).format_data(
                    conf_mat[i, j]
                )
                if float(n) < 1000:
                    n = str(int(float(n)))
                cell_text += n
                if show_normed:
                    cell_text += "\n" + "("
                    cell_text += format(normed_conf_mat[i, j], ".2f") + ")"
            else:
                cell_text += format(normed_conf_mat[i, j], ".2f")
            ax.text(
                x=j,
                y=i,
                s=cell_text,
                va="center",
                ha="center",
                color="white" if normed_conf_mat[i, j] > 0.5 else "black",
            )

    if hide_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.xlabel("predicted label")
    plt.ylabel("true label")
    return fig, ax


def plot_predictions(
    X, model, outdir, name, nrows, ncols, y_val=None, y_pred=None
):
    """Plots a number of predictions for quick inspection"""
    if y_pred is None:
        y_pred = model.predict(X)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(nrows * 2, ncols * 2)
    )
    axes = axes.ravel()

    clrs = ("darkgrey", "salmon", "seagreen", "darkorange", "royalblue", "cyan")
    for i, ax in enumerate(axes):
        xi_val = X[i, :, :]
        yi_prd = y_pred[i, :, :]

        ax.plot(xi_val[:, 0], color="darkgreen", alpha=0.30)
        ax.plot(xi_val[:, 1], color="darkred", alpha=0.30)
        # Plot y_pred as lines
        for j, c in zip(range(len(clrs)), clrs):
            ax.plot(yi_prd[:, j], color=c, lw=2)

        yi_val = y_val[i, :, :] if y_val is not None else y_pred[i, :, :]
        plot_category(yi_val, colors=clrs, alpha=0.30, ax=ax)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_ylim(-0.15, 1.15)
        ax.set_xlim(0, len(xi_val))

    if outdir is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, name + ".pdf"))
        plt.close()


def plot_confusion_matrices(
    y_target,
    y_pred,
    name,
    outdir,
    targets_to_binary=None,
    y_is_binary=False,
    ticks_binary=None,
    ticks_multi=None,
    show_abs=False,
):
    """
    Plots multiclass and binary confusion matrices for smFRET classification
    *Very* hard-coded section, so make sure 0, 1, 2,.. labels match the strings!
    """
    axis = 2 if len(y_target.shape) == 3 else 1
    mkwargs = dict(show_normed=True, show_absolute=show_abs, colorbar=False)

    if y_is_binary:
        matrix = mlxtend.evaluate.confusion_matrix(
            y_target=y_target.argmax(axis=axis).ravel(),
            y_predicted=y_pred.argmax(axis=axis).ravel(),
        )
        fig, ax = _plot_confusion_matrix_mlxtend(matrix, **mkwargs)
        l = (
            ticks_binary
            if ticks_binary is not None
            else ["", "non-usable", "usable"]
        )
        ax.set_yticklabels(l)
        ax.set_xticklabels(l, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, name + "_binary_confusion_matrix.pdf"))
        plt.close()
    else:
        y_target, y_pred = _swap_y_labels(
            *[y.argmax(axis=axis).ravel() for y in (y_target, y_pred)]
        )

        matrix = mlxtend.evaluate.confusion_matrix(
            y_target=y_target, y_predicted=y_pred
        )

        if ticks_multi is not None:
            l = ticks_multi
        else:
            l = [
                "",
                "bleached",
                "aggregate",
                "dynamic",
                "static",
                "noisy",
                "scrambled",
            ]  # default layout

        fig, ax = _plot_confusion_matrix_mlxtend(matrix, **mkwargs)
        ax.set_yticklabels(l)
        ax.set_xticklabels(l, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, name + "_confusion_matrix.pdf"))
        plt.close()

        if (
            targets_to_binary is not None
        ):  # Converts smFRET classification to a binary problem
            y_target_b, y_pred_b = [
                lib.ml.labels_to_binary(
                    y, one_hot=False, to_ones=targets_to_binary
                ).ravel()
                for y in (y_target, y_pred)
            ]
            matrix = mlxtend.evaluate.confusion_matrix(
                y_target=y_target_b, y_predicted=y_pred_b
            )
            l = (
                ticks_binary
                if ticks_binary is not None
                else ["", "non-usable", "usable"]
            )
            fig, ax = _plot_confusion_matrix_mlxtend(matrix, **mkwargs)
            ax.set_yticklabels(l)
            ax.set_xticklabels(l, rotation=90)
            plt.tight_layout()
            plt.savefig(
                os.path.join(outdir, name + "_binary_confusion_matrix.pdf")
            )
            plt.close()


def plot_category(y, ax, colors=None, alpha=0.2):
    """
    Plots a color for every class segment in a timeseries

    Parameters
    ----------
    y_:
        One-hot coded or categorical labels
    ax:
        Ax for plotting
    colors:
        Colors to cycle through
    """
    if colors is None:
        colors = ("darkgrey", "red", "green", "orange", "royalblue", "purple")

    y_ = y.argmax(axis=1) if len(y.shape) != 1 else y
    if len(colors) < len(set(y_)):
        raise ValueError("Must have at least a color for each class")

    adjs, lns = lib.utils.count_adjacent_values(y_)
    position = range(len(y_))
    for idx, ln in zip(adjs, lns):
        label = y_[idx]
        ax.axvspan(
            xmin=position[idx],
            xmax=position[idx] + ln,
            alpha=alpha,
            facecolor=colors[label],
        )
