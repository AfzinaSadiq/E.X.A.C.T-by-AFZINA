from .base_wrapper import BaseWrapper

class TFWrapper(BaseWrapper):
    """Class for wrapping tensorflow models.\n
        Supported Functionalities: \n
        -> predict -> returns the logit/output returned by the model as a numpy object, moved to the cpu.\n
        -> .save -> saves the model's state dictionary to specified path.\n
        -> .load -> loads model weights from specified path for wrapped model.\n
        -> .get_params -> returns model parameters as a Dict object).\n
        -> .set_params -> sets model parameters as per input parameters of type **params\n

        Dependecies = ["tensorflow", "keras"]
    """
    def __init__(self, model, ):
        self.model = model

    def predict(self, X, **kwargs):
        self.model.predict(X, *kwargs)
    
    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        from tensorflow import keras
        self.model = keras.models.load_model(path)

    def get_params(self):
        return {layer.name: layer.get_config() for layer in self.model.layers}
    
    def set_params(self, **params):
        for layer in self.model.layers:
            if layer.name in params:
                layer.set_weights(params[layer.name])

    def get_last_conv_layer(self):
        """
        Finds the last convolutional layer in a TensorFlow/Keras model.\n
        Returns the layer object.
        """
        last_conv = None
        for layer in self.model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv3D)):
                last_conv = layer

        if last_conv is None:
            raise ValueError("No Conv layer found in the TensorFlow model.")

        return last_conv