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
    def __init__(self, model):
        self.model = model


    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)
    
    def predict_proba(self, X, **kwargs):
        import tensorflow as tf
        import numpy as np

        """Return probability outputs"""
        preds = self.model.predict(X, **kwargs)
        preds = np.asarray(preds)
        # self.model.predict() can return different types:a numpy.ndarray (typical for Keras),a tf.Tensor (sometimes with custom models or TF versions),or other array-like objects.
        # np.asarray(preds) guarantees preds is a NumPy array for the rest of the function. 
        # This is important because next you call .ndim, .shape, .min() and .max() which behave as plain NumPy operations — consistent and predictable.

        # Multi-class classification (N,C)
        if preds.ndim == 2 and preds.shape[1]>1:
            # Check if they already look like probabilities
            if (preds.min() < 0) or (preds.max() > 1.0):
                preds = tf.nn.softmax(preds,axis=1).numpy()
        
        #Binary classification (N, 1)
        elif preds.ndim == 2 and preds.shape[1] == 1:
            if (preds.min() < 0) or (preds.max() > 1.0):
                preds = tf.nn.sigmoid(preds).numpy()
        
        # Regression or other types -> return raw preds
        return preds
    
    def save(self, path: str, weights_only:bool = False):
        # If you want only weights:
        if weights_only:
            self.model.save_weights(path)
        else:
            self.model.save(path)

    def load(self, path: str, weights_only:bool = False):
        from tensorflow import keras
        if weights_only:
            self.model.load_weights(path)
        else:
            self.model = keras.models.load_model(path)

    def get_params(self):
        # return {layer.name: layer.get_config() for layer in self.model.layers}
        return {"config": self.model.get_config()}
    
    
    def set_params(self, **params):
        for layer in self.model.layers:
            if layer.name in params:
                layer.set_weights(params[layer.name])

    def get_last_conv_layer(self):
        import tensorflow as tf
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