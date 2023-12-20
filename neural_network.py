from typing import Optional
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn as jnn
import time
import numpy as np

N_LINEAR = 1


@jit
def _predict(params, x):
    """Propagate the regressor through the layers of the network by recursively applying the
    following equation:
        z(l+1) = w(l)x(l) + b(l), x(l+1) = σ(z(l+1)),
        for l ∈ {1, ..., L}
    Here we do not apply the nonlinear activation σ(·) for the last `N_LINEAR` layers.

    Args:
        params: A list of length L containing where each element is a tuple of weights and
                    biases for a given layer [(w_1, b_1), (w_2, b_2), ... (w_L, b_L)].
                    Note: w_1.shape is not necessarily equal to w_2.shape.
        x:      The regressor.

    Returns
        y:      The regressands.
    """
    # With activation
    activations = x  # x(1) = x
    for w, b in params[:-N_LINEAR]:
        outputs = jnp.dot(w, activations) + b  # z(l+1) = w(l)x(l) + b(l)
        activations = jnn.swish(
            outputs
        )  # x(l+1) = σ(z(l+1)), σ being the nonlinear activation

    # Without activation
    for w, b in params[-N_LINEAR:]:
        outputs = jnp.dot(w, activations) + b  # z(l+1) = w(l)x(l) + b(l)
        activations = outputs  # x(l+1) = z(l+1)
    return activations  # return the final x(L), representing the output layer


_batched_predict = vmap(_predict, in_axes=(None, 0))


@jit
def _mse_loss(params, x, y):
    predictions = _batched_predict(params, x)
    return jnp.mean((predictions - y) ** 2)


@jit
def _parameters_gradients(params, x, y):
    gradients = grad(_mse_loss)(params, x, y)
    return gradients


class NeuralNetwork:
    def __init__(self, layers: [int]) -> None:
        """Initialize the neural network with a list of integers.

        Args:
            layers: An L-long list of integers specifying the number of neurons in each layer, with
            L >= 4. The first layer l=0 is the input, the last layer l=(L-1) is the output
            layer. The second-to-last layer l=(L-2) is a layer without a non-linear activation
            function applied to it (allows the network to output values outside the output range of
            activation functions, usually [-1, 1]). Layers l=[1, L-3] are non-linearly activated
            hidden layers.

        Return:
            A `NeuralNetwork` object with randomly initialized network parameters (weights and
            biases).

        Raises:
            AssertionError if len(layers) is less than 4.
        """
        # types_dict = {"tanh": 0, "sigmoid": 1, "relu": 2, "linear": 3}
        # self.layer_types = [types_dict[t] for t in list(zip(*layers))[0]]
        # self.layers = list(zip(*layers))[1]
        assert len(layers) >= N_LINEAR + 2, "Too few layers"
        self.layers = layers
        self.params = self._init_network_params(self.layers, random.PRNGKey(0))

    def _init_network_params(self, sizes, key):
        # Initialize all layers for a fully-connected neural network with sizes "sizes"
        keys = random.split(key, len(sizes))
        return [
            self._random_layer_params(m, n, k)
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)
        ]

    def _random_layer_params(self, m, n, key, scale=1e-2):
        # A helper function to randomly initialize weights and biases
        # for a dense neural network layer
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    def predict(self, X: np.ndarray):
        """Perform the feedforward propagation through the network's layers

        Args:
            X:  A 2D array of samples/regressors (X.shape = (N_samples, sample_dim))

        Returns:
            prediction: A 2D array of predicted regressands (Y.shape = (N_samples, label_dim))

        Raises:
            AssertionError if the dimensions of samples does not match the shape of the first layer
            (X.shape[1] != layers[0]).
        """
        assert X.shape[1] == self.layers[0]
        return _batched_predict(self.params, X)

    def gradients(self, X: np.ndarray, Y: np.ndarray):
        """Compute the backward-propagation of the neural network's output with respect to its
        parameters (weights and biases).

        Args:
            X: A 2D array of samples/regressors (X.shape = (N_samples, sample_dim))
            Y: A 2D array of labels/regressands (Y.shape = (N_samples, label_dim))

        Returns:
            gradients: A list of length L containing where each element is a tuple of weights and
            biases for a given layer [(w_1, b_1), (w_2, b_2), ... (w_L, b_L)]. Note: w_1.shape is
            not necessarily equal to w_2.shape

        Raises:
            AssertionError if the dimension of samples does not match the shape of the first layer
            (X.shape[1] != layers[0]).
            AssertionError if the dimension of labels does not match the shape of the last layer
            (Y.shape[1] != layers[-1]).
        """
        assert X.shape[1] == self.layers[0]
        assert Y.shape[1] == self.layers[-1]
        return _parameters_gradients(self.params, X, Y)

    def fit_grads(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_validation: Optional[np.ndarray] = None,
        Y_validation: Optional[np.ndarray] = None,
        n_epochs: int = 1_000,
        samples_per_batch: int = 1,
        learning_rate: float = 0.01,
    ):
        """Train the neural network using mini-batch gradient descent.

        This method iteratively updates the network's parameters (weights and biases) based on the
        computed gradients from the training data. Optionally, it evaluates the network's performance
        on a validation set after each epoch.

        Args:
            X_train: A 2D array of training samples (shape = (N_samples, sample_dim)).
            Y_train: A 2D array of training labels (shape = (N_samples, label_dim)).
            X_validation: (Optional) A 2D array of validation samples (shape = (N_samples, sample_dim)).
            Y_validation: (Optional) A 2D array of validation labels (shape = (N_samples, label_dim)).
            n_epochs: The number of training epochs (default is 1000).
            samples_per_batch: The number of samples per mini-batch for gradient descent (default is 1).
            learning_rate: The learning rate for gradient descent (default is 0.01).

        Returns:
            A tuple (train_acc, valid_acc) where:
            - train_acc: A list of Mean Squared Error (MSE) loss on the training set for each epoch.
            - valid_acc: A list of MSE loss on the validation set for each epoch (if provided).

        Raises:
            AssertionError: If the dimensions of X_train or Y_train do not match with the network's
            input and output layer dimensions, respectively.
            AssertionError: Similar assertions for X_validation and Y_validation, if provided.
        """
        assert (
            X_train.shape[1] == self.layers[0]
        ), "Input dimension mismatch in training data"
        assert (
            Y_train.shape[1] == self.layers[-1]
        ), "Output dimension mismatch in training data"

        if X_validation is not None and Y_validation is not None:
            assert (
                X_validation.shape[1] == self.layers[0]
            ), "Input dimension mismatch in validation data"
            assert (
                Y_validation.shape[1] == self.layers[-1]
            ), "Output dimension mismatch in validation data"
        else:
            X_validation = X_train.copy()
            Y_validation = Y_train.copy()

        train_acc = []
        valid_acc = []
        # Mini-batch SGD
        try:
            for epoch in range(n_epochs):
                shuffle_indices = np.random.choice(len(X_train), len(X_train))
                X_train = X_train[shuffle_indices]
                Y_train = Y_train[shuffle_indices]
                start_time = time.time()
                for i in range(len(X_train) // samples_per_batch):
                    first_index = i * samples_per_batch
                    indices = range(first_index, first_index + samples_per_batch)
                    grad_weights, grad_biases = list(
                        zip(*self.gradients(X_train[indices], Y_train[indices]))
                    )
                    for layer in range(len(self.params)):
                        current_weight, current_bias = self.params[layer]
                        new_weight = (
                            current_weight - learning_rate * grad_weights[layer]
                        )
                        new_bias = current_bias - learning_rate * grad_biases[layer]
                        self.params[layer] = (new_weight, new_bias)
                epoch_time = time.time() - start_time

                train_acc.append(_mse_loss(self.params, X_train, Y_train))
                valid_acc.append(_mse_loss(self.params, X_validation, Y_validation))
                print(
                    f"Epoch [{epoch + 1:_}/{n_epochs:_}] in {epoch_time:0.6f} sec | "
                    f"Training MSE: {train_acc[-1]:.4e} | Validation MSE: {valid_acc[-1]:.4e}",
                    end="\r",
                )
        except KeyboardInterrupt:
            print("\r")
            print(
                f"Stopped at epoch {epoch + 1:_}/{n_epochs:_}!\n"
                f"Final training MSE: {train_acc[-1]:.4e} | Final validation MSE {valid_acc[-1]:.4e}"
            )
        finally:
            return train_acc, valid_acc


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from scipy.integrate import solve_ivp

    # Lorenz system equations
    def lorenz_system(t, state, sigma=4, rho=50, beta=8 / 3):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    # Generate Lorenz system data
    t_span = [0, 25]
    initial_state = [1, 1, 1]
    t_eval = np.linspace(t_span[0], t_span[1], 10000)
    solution = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t_eval)
    X_train = solution.y.T
    Y_train = np.roll(X_train, -1, axis=0)  # Predicting next state
    # print(f"{X_train.shape = }")
    # print(f"{Y_train.shape = }")

    # Initialize and train the neural network
    nn = NeuralNetwork([3, 16, 3])
    train_acc, valid_acc = nn.fit_grads(
        X_train[:-1],
        Y_train[:-1],
        n_epochs=500,
        samples_per_batch=50,
        learning_rate=1e-4,
    )

    # Predict the trajectory using the neural network
    Y_pred = np.array([nn.predict(x.reshape(1, -1)) for x in X_train[:-1]]).reshape(
        -1, 3
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(train_acc, label="Training MSE")
    ax.plot(valid_acc, label="Validation MSE")
    ax.legend()
    ax.set_title("Mini-batch SGD: Training vs Validation")
    ax.set_ylabel("Mean Squared Error")
    ax.set_xlabel("Epoch")
    plt.show()

    # Animation to compare ground truth with NN prediction
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Setting the axes limits
    ax.set_xlim([-20, 20])
    ax.set_ylim([-30, 30])
    ax.set_zlim([0, 50])

    # Lines for ground truth and prediction
    (line_true,) = ax.plot([], [], [], lw=2, label="Ground Truth", color='cyan', alpha=0.6)
    (line_pred,) = ax.plot(
        [], [], [], lw=0.5, label="NN Prediction", color="indigo", ls="--"
    )
    ax.legend()

    def update(num, X_train, Y_pred, line_true, line_pred):
        line_true.set_data(X_train[:num, :2].T)
        line_true.set_3d_properties(X_train[:num, 2])

        line_pred.set_data(Y_pred[:num, :2].T)
        line_pred.set_3d_properties(Y_pred[:num, 2])

        return line_true, line_pred

    skip_ratio = 10
    ani = FuncAnimation(
        fig,
        update,
        frames=len(X_train) // skip_ratio,
        fargs=(X_train[::skip_ratio], Y_pred[::skip_ratio], line_true, line_pred),
        interval=1,
    )
    plt.show()
