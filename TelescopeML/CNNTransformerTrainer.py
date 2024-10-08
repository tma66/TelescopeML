# ******** Data science / Machine learning Libraries ***************
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    Add,
    GlobalAveragePooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping

tf.get_logger().setLevel("ERROR")

from tensorflow import keras


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_length, d_model):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.pos_encoding = self.positional_encoding(max_length, d_model)

    def positional_encoding(self, length, depth):
        depth = depth / 2.0

        positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

        angle_rates = 1 / (10000**depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)

        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        length = tf.shape(x)[1]
        # Ensure that the positional encoding matches the input length
        pos_encoding = self.pos_encoding[:length, :]
        x = x + pos_encoding[tf.newaxis, :, :]
        return x


class TrainCNNTransformer:
    """
    Train Convolutional Neural Networks with Transformer model

    Parameters
    -----------
    X1_train : array
        Row-StandardScaled input spectra for training.
    X1_val : array
        Row-StandardScaled input spectra for validation.
    X1_test : array
        Row-StandardScaled input spectra for testing.

    X2_train : array
        Col-StandardScaled Mix Max of all rows of input spectra for training.
    X2_val : array
        Col-StandardScaled Mix Max of all rows of input spectra for validation.
    X2_test : array
        Col-StandardScaled Mix Max of all rows of input spectra for testing.

    y1_train : array
        Col-StandardScaled target feature 1 for training.
    y1_val : array
        Col-StandardScaled target feature 1 for validation.
    y1_test : array
        Col-StandardScaled target feature 1 for testing.

    y2_train : array
        Col-StandardScaled target feature 2 for training.
    y2_val : array
        Col-StandardScaled target feature 2 for validation.
    y2_test : array
        Col-StandardScaled target feature 2 for testing.

    y3_train : array
        Col-StandardScaled target feature 3 for training.
    y3_val : array
        Col-StandardScaled target feature 3 for validation.
    y3_test : array
        Col-StandardScaled target feature 3 for testing.

    y4_train : array
        Col-StandardScaled target feature 4 for training.
    y4_val : array
        Col-StandardScaled target feature 4 for validation.
    y4_test : array
        Col-StandardScaled target feature 4 for testing.
    """

    def __init__(
        self,
        X1_train,
        X1_val,
        X1_test,  # Row-StandardScaled input spectra
        X2_train,
        X2_val,
        X2_test,  # Col-StandardScaled Mix Max of all rows of input spetra
        y1_train,
        y1_val,
        y1_test,  # Col-StandardScaled target feature 1
        y2_train,
        y2_val,
        y2_test,  # Col-StandardScaled target feature 2
        y3_train,
        y3_val,
        y3_test,  # Col-StandardScaled target feature 3
        y4_train,
        y4_val,
        y4_test,  # Col-StandardScaled target feature 4
    ):

        # train, val, test sets for input 1 (main 104 spectral features)
        self.X1_train, self.X1_val, self.X1_test = X1_train, X1_val, X1_test

        # train, val, test sets for input 2 (Min and Max 2 features)
        self.X2_train, self.X2_val, self.X2_test = X2_train, X2_val, X2_test

        # train, val, test sets for target features
        self.y1_train, self.y1_val, self.y1_test = y1_train, y1_val, y1_test
        self.y2_train, self.y2_val, self.y2_test = y2_train, y2_val, y2_test
        self.y3_train, self.y3_val, self.y3_test = y3_train, y3_val, y3_test
        self.y4_train, self.y4_val, self.y4_test = y4_train, y4_val, y4_test

    def build_model(
        self,
        config,  # dic
    ):
        """
        Build a CNN model with the given hyperparameters.

        Parameters
        ----------
        hyperparameters : dict
            A dictionary containing hyperparameter settings.

            hyperparameters keys includes:
                - 'Conv__num_blocks' (int): Number of blocks in the CNN model.
                - 'Conv__num_layers_per_block' (int): Number of layers in each convolutional block.
                - 'Conv__num_filters' (list): Number of filters in each layer.
                - 'Conv__kernel_size' (int): Size of the convolutional kernel.
                - 'Conv__MaxPooling1D' (bool): MaxPooling1D size.

                - 'Trans__NumberBlocks' (int): Number of transformer blocks.
                - 'Trans__NumberHeads' (int): Number of heads in the transformer block.
                - 'Trans__dropout' (float): Dropout rate for the transformer block.

                - 'FC1__num_blocks' (int): Number of blocks in the first fully connected (FC1) part.
                - 'FC1_num_layers_per_block' (int): Number of layers in each FC1 block.
                - 'FC1__units' (list): Number of units in each FC1 layer.
                - 'FC1__dropout' (float): Dropout rate for FC1 layers.

                - 'FC2__num_blocks' (int): Number of blocks in the second fully connected (FC2) part.
                - 'FC2_num_layers_per_block' (int): Number of layers in each FC2 block.
                - 'FC2__units' (list): Number of units in each FC2 layer.
                - 'FC2__dropout' (float): Dropout rate for FC2 layers.

                - 'learning_rate' (float): Learning rate for the model.

        Example
        --------
        >>> hyperparameters = {
        >>>      'Conv__MaxPooling1D': 2,
        >>>      'Conv__num_blocks': 1,
        >>>      'Conv__num_layers_per_block': 3,
        >>>      'Conv__num_filters': 4,
        >>>      'Conv__kernel_size': 6,
        >>>      'Trans__NumberBlocks: 1,
        >>>      'Trans__NumberHeads: 4,
        >>>      'Trans__dropout': 0.1,
        >>>      'FC__NumberLayers': 4,
        >>>      'FC1__num_blocks' : 1,
        >>>      'FC1_num_layers_per_block': 4,
        >>>      'FC1__dropout': 0.09889223768186726,
        >>>      'FC1__units': 128,
        >>>      'FC2__num_blocks' : 1,
        >>>      'FC2_num_layers_per_block':2,
        >>>      'FC2__dropout': 0.0024609140719442646,
        >>>      'FC2__units': 64,
        >>>      'learning_rate': 4.9946842008422193e-05}

        Returns
        -------
        object
            Pre-build tf.keras.Model CNN model.

        """

        """
        Convolution Neural Networks to be optimized by BOHB package.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        Conv__filters = config["Conv__filters"]
        Conv__kernel_size = config["Conv__kernel_size"]
        Conv__MaxPooling1D = config["Conv__MaxPooling1D"]
        Conv__NumberLayers = config["Conv__NumberLayers"]
        Conv__NumberBlocks = config["Conv__NumberBlocks"]

        Trans_NumberBlocks = config["Trans__NumberBlocks"]
        Trans__NumberHeads = config["Trans__NumberHeads"]
        Trans__dropout = config["Trans__dropout"]

        FC1__units = config["FC1__units"]
        FC1__dropout = config["FC1__dropout"]
        FC1__NumberLayers = config["FC1__NumberLayers"]

        FC2__units = config["FC2__units"]
        FC2__NumberLayers = config["FC2__NumberLayers"]
        FC2__dropout = config["FC2__dropout"]
        FC2__NumberBlocks = config["FC2__NumberBlocks"]

        lr = config["lr"]
        self.lr = lr

        ######### Shape of the inputs
        input_1 = tf.keras.layers.Input(shape=(104, 1))
        input_2 = tf.keras.layers.Input(shape=(2,))

        ######### Conv Blocks  ####################################
        model = input_1
        for b in range(0, Conv__NumberBlocks):
            for l in range(0, Conv__NumberLayers):
                model = Conv1D(
                    filters=Conv__filters * (b + l + 1) ** 2,
                    kernel_size=Conv__kernel_size,
                    strides=1,
                    padding="same",
                    activation="relu",
                    kernel_initializer="he_normal",
                    # kernel_regularizer=tf.keras.regularizers.l2(Conv__regularizer),
                    name="Conv__B" + str(b + 1) + "_L" + str(l + 1),
                )(
                    model
                )  # (model if l!= 0 and b!= 0 else input_1)

            model = MaxPooling1D(
                pool_size=(Conv__MaxPooling1D),
                name="Conv__B" + str(b + 1) + "__MaxPooling1D",
            )(model)

        # TODO: Apply Positional Embedding if needed
        # pos_embedding = PositionalEmbedding(max_length=104, d_model=model.shape[-1])
        # model = pos_embedding(model)

        ######### Transformer Block #########
        input_dim = model.shape[-1]
        for _ in range(Trans_NumberBlocks):
            # Multi-Head Self Attention Layer
            attn_output = MultiHeadAttention(
                num_heads=Trans__NumberHeads, key_dim=64, dropout=Trans__dropout
            )(model, model)
            attn_output = Dropout(Trans__dropout)(attn_output)
            norm1_output = LayerNormalization(epsilon=1e-6)(attn_output + model)

            # Feedforward Network
            # Feedforward dimension is 2 times the input dimension
            ff_output = Conv1D(input_dim * 2, 1, activation="relu")(norm1_output)
            ff_output = Dropout(Trans__dropout)(ff_output)
            ff_output = Conv1D(input_dim, 1)(
                ff_output
            )  # Project back to original dimension
            # Add and Norm
            model = LayerNormalization(epsilon=1e-6)(ff_output + norm1_output)

        ######### Flatten Layer   ####################################
        model = Flatten()(model)

        ######### FC Layer before the Concatenation   ################
        for l in range(FC1__NumberLayers):
            model = Dense(
                FC1__units * (l + 1) ** 2,
                activation="relu",
                kernel_initializer="he_normal",
                # kernel_regularizer=tf.keras.regularizers.l2(Conv__regularizer),
                name="FC1__B1_L" + str(l + 1),
            )(model)

        model = Dropout(FC2__dropout, name="FC1__B1_L" + str(l + 1) + "__Dropout")(
            model
        )

        ######### Concatenation Layer  ###############################
        # Concatenate the outputs from the convolutional layers and dense layer
        model = tf.keras.layers.concatenate([model, input_2], name="Concatenated_Layer")

        ######### FC Block  ####################################
        for b in range(0, FC2__NumberBlocks):
            for l in range(0, FC2__NumberLayers):
                model = Dense(
                    FC2__units * (b + l + 1) ** 2,
                    activation="relu",
                    kernel_initializer="he_normal",
                    # kernel_regularizer=tf.keras.regularizers.l2(Conv__regularizer),
                    name="FC2__B" + str(b + 1) + "_L" + str(l + 1),
                )(
                    model
                )  # (model if l!= 0 and b!= 0 else input_1)

            model = Dropout(
                FC2__dropout,
                name="FC2__B" + str(b + 1) + "_L" + str(l + 1) + "__Dropout",
            )(model)

        ######### 3rd FC Block: gravity  #############################

        out__gravity = Dense(
            1,
            activation="linear",
            # kernel_initializer = 'he_normal',
            name="output__gravity",
        )(model)

        ######### 3rd FC Block: c_o_ratio  ##############################
        out__c_o_ratio = Dense(
            1,
            activation="linear",
            # kernel_initializer = 'he_normal',
            # kernel_regularizer=tf.keras.regularizers.l2(0.003/2),
            name="output__c_o_ratio",
        )(model)

        ######### 3rd FC Block: metallicity  ##############################

        out__metallicity = Dense(
            1,
            activation="linear",
            # kernel_initializer = 'he_normal',
            name="output__metallicity",
        )(model)

        ######### 3rd FC Block: temperature  ##############################
        out__temperature = Dense(1, activation="linear", name="output__temperature")(
            model
        )

        ######### OUTPUT   ################################################
        # Create the model with two inputs and two outputs
        model = tf.keras.Model(
            inputs=[input_1, input_2],
            outputs=[out__gravity, out__c_o_ratio, out__metallicity, out__temperature],
        )

        self.model = model

        print(model.summary())

    def fit_cnn_model(self, batch_size=32, budget=3):
        """
        Fit the pre-build CNN model

        Returns:
            Training history (Loss values for train)
            Trained model
        """
        model = self.model
        # Compile the model with an optimizer, loss function, and metrics
        model.compile(
            loss="huber_loss",
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["mae"],
        )

        early_stop = EarlyStopping(
            monitor="loss",
            min_delta=4e-4,
            patience=50,
            mode="auto",
            restore_best_weights=True,
        )

        # YOU CAN ADD FUNCTION HERE TO ADD NOISE
        history = self.model.fit(
            x=[self.X1_train, self.X2_train],
            y=[self.y1_train, self.y2_train, self.y3_train, self.y4_train],
            # self.x_train, self.y_train,
            batch_size=batch_size,  # config['batch_size'], # self.batch_size,
            validation_data=(
                [self.X1_val, self.X2_val],
                [self.y1_val, self.y2_val, self.y3_val, self.y4_val],
            ),
            # validation_split=0.2,
            epochs=int(budget),
            verbose=1,
            callbacks=[early_stop],
        )
        self.model = model
        self.history = history
        return history, model
