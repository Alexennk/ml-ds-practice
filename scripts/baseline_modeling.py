import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PredictionVisualizer:
    @staticmethod
    def model_performance_sc_plot(predictions, labels, title="Scatter Plot"):
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
        plt.show()

    @staticmethod
    def plot_residuals(y_true, y_pred, model_name="Model"):
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
        plt.show()

    @staticmethod
    def plot_predictions_distribution(
        y_true, y_pred, histogram=False, model_name="Model"
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
        plt.show()
