from tensorflow import keras
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.layers import (
    Activation,
    Bidirectional,
    Conv1D,
    CuDNNLSTM,
    Dense,
    Dropout,
    MaxPooling1D,
    add,
)
from tensorflow_core.python.keras.layers.normalization_v2 import (
    BatchNormalization,
)
from tensorflow_core.python.keras.layers.recurrent_v2 import LSTM
from tensorflow_core.python.keras.optimizer_v2.gradient_descent import SGD


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

        out = add([res, out])

        return out

    def __call__(self, x):
        return self.build(x)


def create_deepconvlstm_model(google_colab, n_features, regression):
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
    x = ResidualConv1D(filters=32, kernel_size=65, pool=True)(x)
    x = ResidualConv1D(filters=32, kernel_size=65)(x)
    x = ResidualConv1D(filters=32, kernel_size=65)(x)

    x = ResidualConv1D(filters=64, kernel_size=33, pool=True)(x)
    x = ResidualConv1D(filters=64, kernel_size=33)(x)
    x = ResidualConv1D(filters=64, kernel_size=33)(x)

    x = ResidualConv1D(filters=128, kernel_size=15, pool=True)(x)
    x = ResidualConv1D(filters=128, kernel_size=15)(x)
    x = ResidualConv1D(filters=128, kernel_size=15)(x)

    x = ResidualConv1D(filters=256, kernel_size=7, pool=True)(x)
    x = ResidualConv1D(filters=256, kernel_size=7)(x)
    x = ResidualConv1D(filters=256, kernel_size=7)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(1, padding="same")(x)

    x = Bidirectional(LSTM_(16, return_sequences=True))(x)
    final = Dropout(rate=0.4)(x)

    if regression:
        outputs = Dense(1, activation=None)(final)
        metric = "mse"
        loss = "mse"
    else:
        outputs = Dense(6, activation="softmax")(final)
        metric = "accuracy"
        loss = "categorical_crossentropy"

    optimizer = SGD(lr=0.01, momentum=0.8, nesterov=True)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    return model


def get_model(
    n_features,
    train,
    new_model,
    model_name,
    model_path,
    google_colab,
    regression,
    print_summary=True,
    tag=None,
):
    """Loader for model"""
    if train:
        if new_model:
            print("Created new model.")
            model = create_deepconvlstm_model(
                google_colab=google_colab,
                n_features=n_features,
                regression=regression,
            )
        else:
            try:
                if tag is not None:
                    model_name = model_name.replace(
                        "best_model", tag + "_best_model"
                    )
                model = keras.models.load_model(
                    str(model_path.joinpath(model_name))
                )
            except OSError:
                print("No model found. Created new model.")
                model = create_deepconvlstm_model(
                    google_colab=google_colab,
                    n_features=n_features,
                    regression=regression,
                )
    else:
        print("Loading model from file..")
        model = keras.models.load_model(str(model_path.joinpath(model_name)))

    if print_summary:
        model.summary()
    return model
