import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from ..utils import predict_proba_fn


class LimeExplainer_Tabular:
    '''
    LIME Tabular Explainer for Exact Library 

    Responsibilities:
        - Generate LIME explanations for tabular models
        - Return feature importance values
        - Provide optional visualization utilites
    '''

    def __init__(self, model, training_data, feature_names = None, class_names = None, mode = 'classification'):
        '''
        Parameters
        ----------
        model : torch.nn.Module
            Trained PyTorch tabular model

        training_data : np.ndarray
            Training data used by LIME for sampling

        feature_names : list[str], optional
            Names of input features
        
        class_names : list[str], optional
            Output class names

        mode : str
            "classification" or "regression"
        '''

        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode

        self.explainer = LimeTabularExplainer(
            training_data = training_data,
            feature_names=feature_names,
            class_names = class_names,
            mode = mode,
            discretize_continuous=True
        )

    # ---------------------------------------- Core explanation logic ----------------------------------------
    def explain(self, instance, top_labels = 1):
        '''
        Explain a single tabular instance.

        Parameters
        -----------
        instance : np.ndarray
            Shape (num_features,)

        Returns
        --------
        explanation : lime.explanation.Explanation
        '''

        def predict_fn(X):
            return predict_proba_fn.predict_proba(X, model = self.model)
        
        explanation = self.explainer.explain_instance(
            instance,
            predict_fn,
            top_labels=top_labels
        )

        return explanation
    
    # ---------------------------------------- Raw explanation data ----------------------------------------

    def get_explanation_data(self, explanation, label= None, num_features = 10):
        '''
        Get feature importance list.

        Returns
        --------
        list of (feature_name, importance)
        '''
        if label is None:
            label = explanation.top_labels[0]

        return explanation.as_list(label=label)[:num_features]
    
    # ---------------------------------------- Optional visualization ----------------------------------------
    def visualize(self, explanation, label = None, num_features = 10):
        '''
        Print feature importance in readable form.
        '''

        feature_scores = self.get_explanation_data(
            explanation,
            label = label,
            num_features=num_features
        )

        print("\nLIME Tabular Explanation\n"+"-"*30)
        for feature, score in feature_scores:
            sign = "+" if score > 0 else "-"
            print(f"{feature:20s} {sign} {abs(score):.4f}")

        return feature_scores