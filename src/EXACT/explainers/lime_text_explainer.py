import numpy as np
from lime import lime_tabular

class LimeExplainer_Tabular:

    '''

    Lime Tabular Explainer
    Works for both Pytorch and Tensorflow

    '''

    def __init__(self, wrapped_model, training_data, feature_names, class_names = None, categorical_features = None, categorical_names = None, mode = 'classification', num_samples = 3000):
        self.model = wrapped_model
        self.num_samples = num_samples
        
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names = feature_names,
            class_names = class_names,
            categorical_features = categorical_features,
            categorical_names = categorical_names,
            mode = mode,
            discretize_continuous = True
        )

    def explain(self, instance, top_labels = 1):
        """

        Explain single instance
        
        """
        explanation = self.explainer.explain_instance(
            data_row = instance,
            predict_fn = self.model.predict_proba,
            num_features = len(instance),
            top_labels = top_labels,
            num_samples = self.num_samples
        )

        return explanation
    