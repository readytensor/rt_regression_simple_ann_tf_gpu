import os
import warnings
from typing import Callable, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.exceptions import NotFittedError
from tensorflow.keras.callbacks import Callback, EarlyStopping, LambdaCallback
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam

from logger import get_logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")


MODEL_PARAMS_FNAME = "model_params.save"
MODEL_WTS_FNAME = "model_wts.save"
HISTORY_FNAME = "history.json"
COST_THRESHOLD = float("inf")


logger = get_logger(task_name="tf_model_training")

# Check TensorFlow Version
logger.info(f"TensorFlow Version: {tf.__version__}")

# Check for GPU availability
gpu_avai = (
    "GPU available (YES)"
    if tf.config.list_physical_devices("GPU")
    else "GPU not available"
)

logger.info(gpu_avai)


def create_logger(log_period: int, log_type: str = "epoch") -> Callable:
    """
    Create a logging function to log information every log_period epochs or batches.

    This function creates and returns another function:
        `log_function(log_count, logs)`
    which checks if the current log_count number (either epoch or batch number)
    is a multiple of the specified log_period. If it is, it logs the log_count number
    and the logs information.

    Args:
        log_period (int): The period at which to log information. For example, if
                    log_period is 10, the logging will happen at every 10th epoch
                    or batch (e.g., 0th, 10th, 20th, etc.)
        log_type (str): A string that is either 'epoch' or 'batch' specifying the
                    type of logging.
                    Defaults to 'epoch'.

    Returns:
        Callable: The log_function function that logs every log_period epochs
                    or batches.
    """

    def log_function(log_count: int, logs: Dict) -> None:
        logs_str = ""
        for k, v in logs.items():
            logs_str += f"{k}: {np.round(v, 4)}  "
        if log_count % log_period == 0:
            logger.info(f"{log_type.capitalize()}: {log_count}, Metrics: {logs_str}")

    return log_function


class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get("loss")
        if loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val):
            logger.info(f"Cost is {loss_val}, so stopping training!!")
            self.model.stop_training = True


def validate_activation(activation: str) -> Union[str, None]:
    """
    Validate the activation.

    Args:
        activation (str): Name of the activation function.

    Returns:
        Union[str, None]: validated activation.

    Raises:
        ValueError: If the activation string does not match any known
                   activation functions.
    """
    if activation not in ["tanh", "relu", "none", "None", None]:
        raise ValueError(
            f"Error: Unrecognized activation type: {activation}\n"
            "Must be one of ['relu', 'tanh', 'none']"
        )
    if activation in ["none", "None"]:
        activation = None
    return activation


class Regressor:
    """A wrapper class for the ANN regressor in Tensorflow."""

    model_name = "simple_ANN_tensorflow_regressor"

    def __init__(
        self,
        D: Optional[int] = None,
        activation: Optional[str] = "tanh",
        lr: Optional[float] = 1e-3,
        **kwargs,
    ):
        """Construct a new regressor.

        Args:
            D (int, optional): Size of the input layer.
                Defaults to None (set in `fit`).
            activation (str, optional): Activation function for hidden layers.
                Options: ["relu", "tanh", "none"]
                Defaults to "tanh".
            lr (int, optional): Learning rate for optimizer.
                Defaults to 1e-3.
        """
        self.D = D
        self.activation = str(activation).strip()
        self.lr = float(lr)
        self._log_period = 10  # logging per 10 epochs
        # defer building model until fit because we need to know
        # dimensionality of data (D) to define the size of
        # input layer
        self.model = None

    def build_model(self):

        self.activation = validate_activation(self.activation)
        self.D = int(self.D)
        M1 = max(100, int(self.D * 4))
        M2 = max(30, int(self.D * 0.5))

        input_ = Input(self.D)
        x = input_
        x = Dense(M1, activation=self.activation)(x)
        x = Dense(M2, activation=self.activation)(x)
        x = Dense(1)(x)
        output_ = x
        model = Model(input_, output_)
        # model.summary()
        model.compile(
            loss='mse',
            # optimizer=SGD(learning_rate=self.lr),
            optimizer=Adam(learning_rate=self.lr),
            metrics=[tfa.metrics.RSquare(dtype=tf.float32)]
        )
        return model

    def fit(
        self,
        train_inputs: pd.DataFrame,
        train_targets: pd.Series,
        batch_size=100,
        epochs=1000,
    ) -> None:
        """Fit the regressor to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        # get data dimensionality and build network
        self.D = train_inputs.shape[1]
        self.model = self.build_model()

        # set seed for reproducibility
        tf.random.set_seed(0)

        # use 15% validation split if at least 300 samples in training data
        if train_inputs.shape[0] < 300:
            loss_to_monitor = "loss"
            validation_split = None
        else:
            loss_to_monitor = "val_loss"
            validation_split = 0.15

        early_stop_callback = EarlyStopping(
            monitor=loss_to_monitor, min_delta=1e-3, patience=30
        )
        infcost_stop_callback = InfCostStopCallback()
        logger_callback = LambdaCallback(
            on_epoch_end=create_logger(self._log_period, "epoch")
        )

        self.model.fit(
            x=train_inputs,
            y=train_targets,
            batch_size=batch_size,
            validation_split=validation_split,
            epochs=epochs,
            shuffle=True,
            verbose=True,
            callbacks=[
                early_stop_callback,
                infcost_stop_callback,
                logger_callback,
            ],
        )

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict targets for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted targets.
        """
        preds = np.squeeze(self.model.predict(inputs, verbose=True))
        # Check if the prediction is a scalar
        if np.ndim(preds) == 0:
            preds = np.reshape(preds, [1])
        return preds

    def summary(self):
        """Return model summary of the Tensorflow model"""
        self.model.summary()

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the model and return the loss and metrics

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The mse of the regressor.
        """
        if self.model is not None:
            # returns list containing loss value and metric value
            # index at 1 which contains mse
            return self.model.evaluate(test_inputs, test_targets, verbose=0)[1]
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the regressor to disk.

        Args:
            model_dir_path (str): The dir path to which to save the model.
        """
        if self.model is None:
            raise NotFittedError("Model is not fitted yet.")
        model_params = {
            "D": self.D,
            "activation": self.activation,
            "lr": self.lr,
        }
        joblib.dump(model_params, os.path.join(model_dir_path, MODEL_PARAMS_FNAME))
        self.model.save_weights(os.path.join(model_dir_path, MODEL_WTS_FNAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Regressor: A new instance of the loaded regressor.
        """
        if not os.path.exists(model_dir_path):
            raise FileNotFoundError(f"Model dir {model_dir_path} does not exist.")
        model_params = joblib.load(os.path.join(model_dir_path, MODEL_PARAMS_FNAME))
        regressor_model = cls(**model_params)
        regressor_model.model = regressor_model.build_model()
        regressor_model.model.load_weights(
            os.path.join(model_dir_path, MODEL_WTS_FNAME)
        ).expect_partial()
        return regressor_model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"activation: {self.activation}, "
            f"D: {self.D}, "
            f"lr: {self.lr})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Regressor:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data targets.
        hyperparameters (dict): Hyperparameters for the regressor.

    Returns:
        'Regressor': The regressor model
    """
    regressor = Regressor(**hyperparameters)
    regressor.fit(train_inputs=train_inputs, train_targets=train_targets)
    return regressor


def predict_with_model(regressor: Regressor, data: pd.DataFrame) -> np.ndarray:
    """
    Predict regression targets for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted regression targets.
    """
    return regressor.predict(data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Regressor, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the regressor model and return the r-squared value.

    Args:
        model (Regressor): The regressor model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The targets of the test data.

    Returns:
        float: The r-sq value of the regressor model.
    """
    return model.evaluate(x_test, y_test)


def save_training_history(history, dir_path):
    """
    Save tensorflow model training history to a JSON file
    """
    hist_df = pd.DataFrame(history.history)
    hist_json_file = os.path.join(dir_path, HISTORY_FNAME)
    with open(hist_json_file, mode="w", encoding="utf-8") as file_:
        hist_df.to_json(file_)
