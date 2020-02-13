import sys
import matplotlib.pyplot as plt
import keras
import tensorflow
import sklearn.model_selection
from keras.layers import *
from pathlib import Path

GOOGLE_COLAB = "google.colab" in sys.modules
if GOOGLE_COLAB:
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


def data(datadir):
    X, y_label = lib.utils.load_npz_data(
        top_percentage=PERCENT,
        path=datadir,
        set_names=("X_" + DATANAME, "y_" + DATANAME),
    )
    set_y = set(y_label.ravel())
    print(X.shape)
    print(set_y)

    y = lib.ml.class_to_one_hot(y_label, num_classes=len(set_y))
    X = lib.utils.sample_max_normalize_3d(X)
    print("X: ", X.shape)
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    return X_train, X_test, y_train, y_test


class ResidualConv1D:
    """
    ResidualConv1D for use with best performing classifier
    """

    def __init__(self, filters, kernel_size, pool=False):
        self.pool = pool
        self.kernel_size = kernel_size
        self.params = {
            "padding": "same",
            "kernel_initializer": "he_uniform",
            "strides": 1,
            "filters": filters,
        }

    def build(self, x):

        res = x
        if self.pool:
            x = MaxPooling1D(1, padding="same")(x)
            res = Conv1D(kernel_size=1, **self.params)(res)

        out = Conv1D(kernel_size=1, **self.params)(x)

        out = BatchNormalization()(out)
        out = Activation("relu")(out)
        out = Conv1D(kernel_size=self.kernel_size, **self.params)(out)

        out = BatchNormalization()(out)
        out = Activation("relu")(out)
        out = Conv1D(kernel_size=self.kernel_size, **self.params)(out)

        out = keras.layers.add([res, out])

        return out

    def __call__(self, x):
        return self.build(x)


def create_model(google_colab, n_features):
    """
    Creates Keras model that resulted in the best performing classifier so far
    """

    LSTM_ = CuDNNLSTM if google_colab else LSTM

    inputs = Input(shape=(None, n_features))

    x = Conv1D(
        filters=32,
        kernel_size=16,
        padding="same",
        kernel_initializer="he_uniform",
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # residual net part
    x = ResidualConv1D(filters=32, kernel_size=16, pool=True)(x)
    x = ResidualConv1D(filters=32, kernel_size=16)(x)
    x = ResidualConv1D(filters=32, kernel_size=16)(x)

    x = ResidualConv1D(filters=64, kernel_size=12, pool=True)(x)
    x = ResidualConv1D(filters=64, kernel_size=12)(x)
    x = ResidualConv1D(filters=64, kernel_size=12)(x)

    x = ResidualConv1D(filters=128, kernel_size=8, pool=True)(x)
    x = ResidualConv1D(filters=128, kernel_size=8)(x)
    x = ResidualConv1D(filters=128, kernel_size=8)(x)

    x = ResidualConv1D(filters=256, kernel_size=4, pool=True)(x)
    x = ResidualConv1D(filters=256, kernel_size=4)(x)
    x = ResidualConv1D(filters=256, kernel_size=4)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(1, padding="same")(x)

    x = Bidirectional(LSTM_(16, return_sequences=True))(x)
    final = Dropout(rate=0.4)(x)

    activation = "softmax"
    loss = "categorical_crossentropy"
    acc = "accuracy"

    outputs = Dense(6, activation=activation)(final)
    optimizer = keras.optimizers.sgd(lr=0.01, momentum=0.8, nesterov=True)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=[acc])
    return model


def get_model(
    n_features,
    train,
    new_model,
    model_name,
    model_path,
    google_colab,
    print_summary=True,
):
    """Loader for model"""
    if train:
        if new_model:
            print("Created new model.")
            model = create_model(
                google_colab=google_colab, n_features=n_features
            )
        else:
            try:
                if TAG is not None:
                    model_name = model_name.replace(
                        "best_model", TAG + "_best_model"
                    )
                model = keras.models.load_model(
                    str(model_path.joinpath(model_name))
                )
            except OSError:
                print("No model found. Created new model.")
                model = create_model(
                    google_colab=google_colab, n_features=n_features
                )
    else:
        print("Loading model from file..")
        model = keras.models.load_model(str(model_path.joinpath(model_name)))

    if print_summary:
        model.summary()
    return model


def main():
    global DATANAME
    global ROOTDIR

    model_name = "{}_best_model.h5".format(DATANAME)
    if GOOGLE_COLAB:
        ROOTDIR = (
            "./gdrive/My Drive/Colab Notebooks"
            + str(ROOTDIR).split("Colab Notebooks")[-1]
        )

    rootdir = Path(ROOTDIR)
    datadir = rootdir.joinpath(DATADIR).expanduser()
    outdir = rootdir.joinpath(OUTDIR).expanduser()

    X_train, X_test, y_train, y_test = data(datadir=datadir)

    model = get_model(
        n_features=X_train.shape[-1],
        train=TRAIN,
        new_model=NEW_MODEL,
        model_name=model_name,
        model_path=outdir,
        google_colab=GOOGLE_COLAB,
    )

    if TAG is not None:
        DATANAME += "_" + TAG
        model_name = model_name.replace("best_model", TAG + "_best_model")

    if TRAIN:
        callbacks = lib.ml.generate_callbacks(
            patience=CALLBACK_TIMEOUT, outdir=outdir, name=DATANAME
        )
        model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
        )
        lib.plotting.plot_losses(logpath=outdir, outdir=outdir, name=DATANAME)

        if GOOGLE_COLAB:
            print("Converted model from GPU to CPU-compatible")
            cpu_model = create_model(
                google_colab=False, n_features=X_train.shape[-1]
            )
            lib.ml.gpu_model_to_cpu(
                trained_gpu_model=model,
                untrained_cpu_model=cpu_model,
                outdir=outdir,
                modelname=model_name,
            )

    print("Evaluating...")
    y_pred = model.predict(X_test)
    lib.plotting.plot_confusion_matrices(
        y_target=y_test,
        y_pred=y_pred,
        y_is_binary=False,
        targets_to_binary=[2, 3],
        outdir=outdir,
        name=DATANAME,
    )


if __name__ == "__main__":
    # In order to run this on Google Colab, everything must be placed
    # according to "~/Google Drive/Colab Notebooks/DeepFRET/"
    ROOTDIR = "."

    # Suffix of the data name, separated by underscore (no need to write X, y)
    DATANAME = "sim_v3"

    # Applies name tag to model name, if any is given
    TAG = None

    DATADIR = "data"
    OUTDIR = "output"
    TRAIN = True
    NEW_MODEL = True

    # Percentage of data to use
    PERCENT = 100
    # Set to None for variable length traces (not supported for all
    # data/model setups)
    N_TIMESTEPS = None

    BATCH_SIZE = 128
    CALLBACK_TIMEOUT = 3
    EPOCHS = 1

    main()
