import sys
import matplotlib.pyplot as plt
import keras
import tensorflow
import sklearn.model_selection
from keras.layers import *
from pathlib import Path

google_colab = "google.colab" in sys.modules
if google_colab:
    sys.path.append("./gdrive/My Drive/Colab Notebooks/DeepFRET-Model")
    plt.style.use("default")
    config = tensorflow.ConfigProto(device_count={"GPU": 1})
    keras.backend.set_session(tensorflow.Session(config=config))
else:
    config = tensorflow.ConfigProto(
        intra_op_parallelism_threads=8, inter_op_parallelism_threads=8
    )
    keras.backend.tensorflow_backend.set_session(
        tensorflow.Session(config=config)
    )

import lib.plotting
import lib.ml
import lib.utils


def main(
    running_on_google_colab,
    datadir,
    rootdir,
    outdir,
    percent_of_data,
    regression,
    dataname,
    tag,
    train,
    new_model,
    callback_timeout,
    epochs,
    batch_size,
):

    if running_on_google_colab:
        rootdir = (
            "./gdrive/My Drive/Colab Notebooks"
            + str(rootdir).split("Colab Notebooks")[-1]
        )

    rootdir = Path(rootdir)
    outdir = rootdir.joinpath(outdir).expanduser()
    datadir = rootdir.joinpath(datadir).expanduser()

    X, labels = lib.utils.load_npz_data(
        top_percentage=percent_of_data,
        path=datadir,
        set_names=("X_" + dataname, "y_" + dataname),
    )

    if not regression:
        # Use labels as classification target
        set_y = set(labels.ravel())
        print(X.shape)
        print(set_y)
        y = lib.ml.class_to_one_hot(labels, num_classes=len(set_y))
        y = lib.ml.smoothe_one_hot_labels(y, amount=0.1)
        X = X[..., 0:3]
    else:
        # Use E_true column as regression target
        y = X[..., 3]
        X = X[..., 0:3]

    X = lib.utils.sample_max_normalize_3d(X)
    print("X: ", X.shape)
    print("Splitting dataset...")
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    model_name = "{}_best_model.h5".format(dataname)

    model = get_model(
        n_features=X_train.shape[-1],
        train=train,
        new_model=new_model,
        model_name=model_name,
        model_path=outdir,
        google_colab=running_on_google_colab,
        tag=tag,
        regression=regression,
    )

    if tag is not None:
        dataname += "_" + tag
        model_name = model_name.replace("best_model", tag + "_best_model")

    if train:
        callbacks = lib.ml.generate_callbacks(
            patience=callback_timeout, outdir=outdir, name=dataname
        )
        model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )
        lib.plotting.plot_losses(logpath=outdir, outdir=outdir, name=dataname)

        if running_on_google_colab:
            print("Converted model from GPU to CPU-compatible")
            cpu_model = create_deepconvlstm_model(
                google_colab=False,
                n_features=X_train.shape[-1],
                regression=regression,
            )
            lib.ml.gpu_model_to_cpu(
                trained_gpu_model=model,
                untrained_cpu_model=cpu_model,
                outdir=outdir,
                modelname=model_name,
            )

    print("Evaluating...")
    y_pred = model.predict(X_val)

    if not regression:
        lib.plotting.plot_confusion_matrices(
            y_target=y_val,
            y_pred=y_pred,
            y_is_binary=False,
            targets_to_binary=[2, 3],
            outdir=outdir,
            name=dataname,
        )


if __name__ == "__main__":
    # In order to run this on Google Colab, everything must be placed
    # according to "~/Google Drive/Colab Notebooks/DeepFRET/"
    main(
        running_on_google_colab=google_colab,
        regression=True,
        train=True,
        new_model=True,
        rootdir=".",
        datadir="data",
        outdir="output",
        dataname="sim",
        tag="experimental",
        percent_of_data=100,
        batch_size=32,
        epochs=10,
        callback_timeout=3,
    )

    """
    Description:
    - Changed kernel sizes to be uneven (should improve convolutions and therefore accuracy?)
    - Very large kernel sizes
    - False negatives significantly lowered, while false positives slightly increased over model 9
    - Label smoothing with amount = 0.05 (0.1 aggressively lowered confidence)
    - Simulated new data with more states, 2 kinds of noise, much longer dark 
    states, more traces
    """
