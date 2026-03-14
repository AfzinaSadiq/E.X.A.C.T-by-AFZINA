import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from EXACT.utils import predict_proba_fn
import matplotlib.pyplot as plt
import os


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
            return predict_proba_fn.predict_proba(X, model = self.model, mode = self.mode)
        
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

        return feature_scores\
        
    # ----------------------------------------
    # Bar plot visualization (NEW ADDITION)
    # ----------------------------------------
    def plot_explanation(self, explanation, label=None, num_features=10,
                         figsize=(8,5), title="LIME Feature Contributions",
                         show=True, save_png=None):
        '''
        Plot feature contributions as a horizontal bar chart.

        Parameters
        ----------
        explanation : lime.explanation.Explanation

        label : int, optional
            Class label to visualize

        num_features : int
            Number of features to display

        figsize : tuple
            Figure size

        title : str
            Plot title

        show : bool
            Whether to display the plot

        save_png : bool
            Save visualization to user_saves directory
        '''

        feature_scores = self.get_explanation_data(
            explanation,
            label=label,
            num_features=num_features
        )

        # Separate features and scores
        features = [f for f, s in feature_scores]
        scores = [s for f, s in feature_scores]

        # Green for positive, red for negative contributions
        colors = ["green" if s > 0 else "red" for s in scores]

        plt.figure(figsize=figsize)

        plt.barh(features, scores, color=colors)

        plt.xlabel("Contribution Weight")
        plt.title(title)

        # Vertical line at zero
        plt.axvline(0)

        # Highest importance on top
        plt.gca().invert_yaxis()

        # Save image
        if save_png:
            save_dir = "user_saves"
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, "lime_tabular_explanation.png")
            plt.savefig(save_path, bbox_inches="tight")

            print(f"LIME tabular visualization saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return features, scores