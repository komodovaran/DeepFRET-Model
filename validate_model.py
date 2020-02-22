import tensorflow.keras.models
import lib.ml
import lib.utils
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt

REGRESSION = True

X, labels = lib.utils.load_npz_data(
    top_percentage=100,
    path="./data",
    set_names=("X_" + "sim", "y_" + "sim"),
)
print(X.shape)

print("Contains labels: ", np.unique(labels))

if not REGRESSION:
    # Use labels as classification target
    set_y = set(labels.ravel())
    y = lib.ml.class_to_one_hot(labels, num_classes=len(set_y))
    y = lib.ml.smoothe_one_hot_labels(y, amount=0.1)
else:
    # Use E_true column as regression target
    y = np.expand_dims(X[..., 4], axis = -1)

print("X: ", X.shape)
print("y: ", y.shape)

print("Splitting dataset...")
_, X_val, _, y_val = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2, random_state=1
)

model = tensorflow.keras.models.load_model(
    "output/sim_experimental_best_model.h5")

E_true = X_val[..., 3]
E_pred = model.predict(np.expand_dims(X_val[0:50, :, 4], axis = -1))

fig, ax = plt.subplots(nrows = 5, ncols = 2, figsize = (15, 10))

for i in range(5):
    ax[i, 0].plot(X_val[i, :, 0], color = "green", label = "D")
    ax[i, 0].plot(X_val[i, :, 1], color = "red", label = "A")

    ax[i, 1].plot(E_true[i], label = "FRET", color = "grey")
    ax[i, 1].plot(E_pred[i], label = "FRET PRED", color = "red")
    ax[i, 1].plot(y_val[i], label = "FRET TRUE", color = "orange")
    ax[i, 1].set_ylim(0, 1)

    ax[i, 0].legend(loc = "upper right")
    ax[i, 1].legend(loc = "upper right")
plt.tight_layout()
plt.show()