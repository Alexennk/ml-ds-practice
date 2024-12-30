import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile


class PredictionVisualizer:
    @staticmethod
    def model_performance_sc_plot(
        predictions, labels, title="Scatter Plot", for_neptune=False
    ):
        min_val = max(max(predictions), max(labels))
        max_val = min(min(predictions), min(labels))
        performance_df = pd.DataFrame({"Label": labels})
        performance_df["Prediction"] = predictions
        sns.jointplot(
            y="Label",
            x="Prediction",
            data=performance_df,
            kind="reg",
            height=6,
            color="blue",
            joint_kws={"scatter_kws": dict(alpha=0.05)},
        )
        plt.plot([min_val, max_val], [min_val, max_val], "m--")
        plt.suptitle(title, fontsize=16)
        if for_neptune:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name)
                plt.close()

                # Log the plot to Neptune
                return tmpfile.name
        else:
            plt.show()

    @staticmethod
    def plot_residuals(y_true, y_pred, model_name="Model", for_neptune=False):
        residuals = y_true - y_pred
        _, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].scatter(y_pred, residuals, alpha=0.05)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title(f"{model_name} - Residual Plot")

        sns.histplot(residuals, color="red", ax=axes[1])
        axes[1].set_xlabel("Residuals")
        axes[1].set_title(f"{model_name} - Residual Distribution")
        plt.tight_layout()

        if for_neptune:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name)
                plt.close()

                # Log the plot to Neptune
                return tmpfile.name
        else:
            plt.show()

    @staticmethod
    def plot_predictions_distribution(
        y_true, y_pred, histogram=False, model_name="Model", for_neptune=False
    ):
        plt.figure(figsize=(6, 4))

        if histogram:
            sns.histplot(y_true, bins=20, label="True target")
            sns.histplot(y_pred, bins=20, label="Predicted target")
            title = "Histograms"
        else:
            sns.kdeplot(y_true, label="True target", bw_adjust=1.5)
            sns.kdeplot(y_pred, label="Predicted target", bw_adjust=1.5)
            title = "Density plots"

        plt.title(f"{title} for {model_name}", fontsize=14)
        plt.xlabel("item_cnt_month", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.ylim(0, None)
        plt.xlim(0, 25)
        plt.xticks(range(-2, 22))
        plt.legend(title="Target sets", fontsize=10)

        if for_neptune:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name)
                plt.close()

                # Log the plot to Neptune
                return tmpfile.name
        else:
            plt.show()

    @staticmethod
    def plot_coefficients(model, df, model_name="Lasso"):
        non_zero_coeffs = model.coef_[model.coef_ != 0]
        non_zero_features = df.columns[model.coef_ != 0]

        sorted_indices = np.argsort(non_zero_coeffs)
        sorted_coeffs = non_zero_coeffs[sorted_indices]
        sorted_features = non_zero_features[sorted_indices]

        plt.figure(figsize=(8, 4))
        plt.barh(range(len(sorted_coeffs)), sorted_coeffs, color="skyblue")
        plt.yticks(range(len(sorted_coeffs)), sorted_features)
        plt.xlabel("Coefficient Value")
        plt.title(model_name + " Coefficients (Non-Zero)")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_importances(model, df, model_name="Tuned Random Forest"):
        feature_importances = model.feature_importances_
        features = df.columns

        importance_df = pd.DataFrame(
            {"Feature": features, "Importance": feature_importances}
        )
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 7))
        plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Feature Importances ({model_name})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
