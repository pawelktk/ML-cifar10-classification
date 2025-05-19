import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import json

def plot_learning_curves(history, model_name):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train acc')
    plt.plot(history.history['val_accuracy'], label='Val acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.title('Loss')
    plt.legend()

    plt.savefig(f"results/{model_name}_learning_curve.png")
    plt.close()



def evaluate_model(model, x_test, y_test, model_name, training_time, training_config):
    y_pred = model.predict(x_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    report = classification_report(y_true, y_pred_classes, output_dict=True)
    test_metrics = model.evaluate(x_test, y_test, verbose=0)
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    cm_path = f"results/{model_name}_confusion_matrix.png"
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(cm_path)
    plt.close()

    results_path = f"results/{model_name}_metrics.json"
    data = {
        "name": model_name.replace('_', ' ').title(),
        "accuracy": round(test_metrics[1], 4),
        "loss": round(test_metrics[0], 4),
        "precision": round(test_metrics[2], 4),
        "recall": round(test_metrics[3], 4),
        "training_time": round(training_time, 1),
        "training_config": str(training_config),
        "learning_curve": f"results/{model_name}_learning_curve.png",
        "confusion_matrix": cm_path,
        "architecture": getattr(model, 'architecture_description', "TODO"),
        "classification_report": report
    }
    
    with open(results_path, "w") as f:
        json.dump(data, f, indent=4)
