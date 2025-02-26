import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from preprocessor import CensusPreprocessor
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.utils import class_weight
from sklearn.inspection import permutation_importance

SUPPORTED_MODELS = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier", "XGBRFClassifier", "GradientBoostingClassifier"]
COLOR_0 = "#4DC9C3"
COLOR_1 = "#221C35"

class CensusTrainer:
    def __init__(self, train_df: pd.DataFrame, train_config: dict):
        self.train_df = train_df
        self.config = train_config
        self.experiment_name = train_config.get("experiment_name")
        self.train_df_processed = None
        self.valid_df_processed = None
        self.train_features = None
        self.classifier = None
        self.class_weights = None
        self.permutation_importances_df = None

    def preprocess_data(self):
        """
        I generally love X-fold validation, and will add that in if time permits (though here we're not really
        struggling with too little data, so not critical)
        """
        preprocessing_config = self.config["preprocessing"]
        preprosessor = CensusPreprocessor(preprocessing_config)
        df_processed = preprosessor.prepare_dataframe(self.train_df)
        self.train_features = list(df_processed.columns)

        df_processed["y"] = (self.train_df["income_details"] == " 50000+.") * 1  # Hard coding for this use case

        # I know sklearn has TT-split built in, but prefer to do it within the DataFrames
        np.random.seed(14)  # Could set in config
        df_processed["set"] = pd.Series((np.random.rand(len(df_processed)) > 0.8) * 1)
        self.train_df_processed = df_processed[df_processed["set"] == 0].reset_index(drop=True).drop(columns="set")
        self.valid_df_processed = df_processed[df_processed["set"] == 1].reset_index(drop=True).drop(columns="set")
        
        # Doing train class weights more manually so it can work with XGBoost, which only has it as a .fit() argument
        self.class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=self.train_df_processed['y'])
        

    def train_classifier(self):
        train_config = self.config["train"]
        if train_config.get("classifier") not in SUPPORTED_MODELS:
            print(f"Please specify a valid 'classifier' -- {str(SUPPORTED_MODELS)}")
            sys.exit()

        cls_name = train_config.pop("classifier")
        # Some non-negotiables I'll always want to fix:
        train_config |= {"random_state": 14,
                         "n_jobs": 10}
        if cls_name == "GradientBoostingClassifier":
            _ = train_config.pop("n_jobs")  # last minute hack... why, GBC!???
        self.classifier = globals()[cls_name](**train_config)

        self.classifier.fit(self.train_df_processed.drop(columns="y").values,
                            self.train_df_processed["y"].values,
                            sample_weight=self.class_weights)

    @staticmethod
    def print_roc_curve(x, y_valid, y_train=None) -> None:
        main_plot_label = "validation" if y_train is not None else "test"
        plt.plot(x, y_valid, label=main_plot_label, color=COLOR_0)
        if y_train is not None:
            plt.plot(x, y_train, label="train", color=COLOR_1)
        plt.plot([0, 1], [0, 1], '--r')
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend()
        plt.title(f"AUC: {round(auc(x, y_valid), 3)}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    def evaluate_classifier(self) -> None:
        try:
            true_y = self.valid_df_processed["y"]
            pred_y = self.classifier.predict(self.valid_df_processed.drop(columns="y").values)
            pred_proba_y = self.classifier.predict_proba(self.valid_df_processed.drop(columns="y").values)[:, 1]
            true_train_y = self.train_df_processed["y"]
            pred_proba_train_y = self.classifier.predict_proba(self.train_df_processed.drop(columns="y").values)[:, 1]

            report = classification_report(true_y, pred_y)
            print(report)

            x, y, _ = roc_curve(true_y, pred_proba_y)
            xt, yt, _ = roc_curve(true_train_y, pred_proba_train_y)
            yt_interpolated = np.interp(x, xt, yt)
            print(f"AUC score: {round(auc(x, y), 3)}")
            self.print_roc_curve(x, y, yt_interpolated)

        except NotFittedError as e:
            print(repr(e))

    @staticmethod
    def plot_feature_importances(importances_df, title=None) -> None:
        plt.barh(importances_df["feature_name"].values,
                 importances_df["feature_importance_mean"].values,
                 yerr=importances_df["feature_importance_std"].values,
                 capsize=5,
                 color=COLOR_0)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        if title:
            plt.title(title)
        plt.show()


    def get_feature_importances(self, title=None) -> None:
        importances_df = (
            pd.DataFrame({"feature_name": self.train_features,
                          "feature_importance_mean": self.classifier.feature_importances_,
                          "feature_importance_std": [0] * len(self.train_features)})
            .sort_values(by="feature_importance_mean", ascending=True)
        )
        self.plot_feature_importances(importances_df, title=title)

    def get_permutation_feature_importances(self, title=None) -> None:
        importances = permutation_importance(self.classifier,
                                             self.train_df_processed.drop(columns="y").values,
                                             self.train_df_processed["y"].values,
                                             random_state=14)
        self.permutation_importances_df = (
            pd.DataFrame({"feature_name": self.train_features,
                          "feature_importance_mean": importances["importances_mean"],
                          "feature_importance_std": importances["importances_std"]})
            .sort_values(by="feature_importance_mean", ascending=True)
        )

        self.plot_feature_importances(self.permutation_importances_df, title=title)
        
    def evaluate_on_unseen_dataset(self, test_df: pd.DataFrame) -> None:
        preprocessing_config = self.config["preprocessing"]  # could refactor code duplication
        preprosessor = CensusPreprocessor(preprocessing_config)
        df_processed = preprosessor.prepare_dataframe(test_df)
        assert self.train_features == list(df_processed.columns)  # Fingers-crossed moment :-(
        
        df_processed["y"] = (test_df["income_details"] == " 50000+.") * 1  # Hard coding again
        
        # Again, this whole next bit could be factored away, but I really haven't the time right now
        try:
            true_y = df_processed["y"]
            pred_y = self.classifier.predict(df_processed.drop(columns="y").values)
            pred_proba_y = self.classifier.predict_proba(df_processed.drop(columns="y").values)[:, 1]

            report = classification_report(true_y, pred_y)
            print(report)

            x, y, _ = roc_curve(true_y, pred_proba_y)
            print(f"Test AUC score: {round(auc(x, y), 3)}")
            self.print_roc_curve(x, y)

        except NotFittedError as e:
            print(repr(e))
            