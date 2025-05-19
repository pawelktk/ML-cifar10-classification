import tensorflow as tf
import time
import os
import json
from datetime import datetime
from src.preprocessing import load_and_preprocess_data
from src.models import (
    create_simple_cnn_model,
    create_dropout_cnn_model,
    create_batchnorm_cnn_model,
    create_deep_cnn_model,
    create_mobilenet_transfer_model
)
from src.utils import plot_learning_curves, evaluate_model
from src.report import generate_html_report, collect_results

def get_model_config(model_name):
    """Zwraca konfigurację treningową dla danego modelu"""
    configs = {
        'simple_cnn': {
            'epochs': 15,
            'batch_size': 64,
            'learning_rate': 0.001,
            'early_stopping': True,
            'reduce_lr': False
        },
        'dropout_cnn': {
            'epochs': 20,
            'batch_size': 64,
            'learning_rate': 0.001,
            'early_stopping': True,
            'reduce_lr': True
        },
        'batchnorm_cnn': {
            'epochs': 25,
            'batch_size': 128,
            'learning_rate': 0.002,
            'early_stopping': True,
            'reduce_lr': True
        },
        'deep_cnn': {
            'epochs': 30,
            'batch_size': 64,
            'learning_rate': 0.0005,
            'early_stopping': True,
            'reduce_lr': True
        },
        'mobilenet_transfer': {
            'epochs': 15,
            'batch_size': 32,
            'learning_rate': 0.0001,
            'early_stopping': True,
            'reduce_lr': True
        }
    }
    return configs.get(model_name, {
        'epochs': 20,
        'batch_size': 64,
        'learning_rate': 0.001,
        'early_stopping': True,
        'reduce_lr': True
    })

def train_and_evaluate(model_fn, model_name, x_train, y_train, x_test, y_test):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    config = get_model_config(model_name)
    
    print(f"\nTraining {model_name} with config: {config}")
    model = model_fn(x_train.shape[1:], 10)
    
    callbacks = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if config['early_stopping']:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        )
    
    if config['reduce_lr']:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        )
    
    model_path = f"models/{model_name}_{timestamp}.h5"
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor='val_accuracy'
        )
    )
    
    log_dir = f"logs/{model_name}_{timestamp}"
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    
    final_model_path = f"models/{model_name}.h5"
    model.save(final_model_path)
    
    plot_learning_curves(history, model_name)
    evaluate_model(model, x_test, y_test, model_name, training_time, config)
    
    return history

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_and_preprocess_data()
    
    models_to_train = [
        (create_simple_cnn_model, "simple_cnn"),
        (create_dropout_cnn_model, "dropout_cnn"),
        (create_batchnorm_cnn_model, "batchnorm_cnn"),
        (create_deep_cnn_model, "deep_cnn"),
        (create_mobilenet_transfer_model, "mobilenet_transfer")
    ]
    
    for model_fn, model_name in models_to_train:
        train_and_evaluate(
            model_fn, model_name, 
            x_train, y_train, 
            x_test, y_test
        )
    
    model_results = collect_results()
    best_model = model_results[0]
    
    explanation = (
        f"Model {best_model['name']} osiągnął najlepszą dokładność ({best_model['accuracy']:.2%}) "
        f"przy czasie trenowania {best_model['training_time']:.1f}s. "
        f"Użyta konfiguracja: {best_model['training_config']}."
    )
    
    generate_html_report(model_results, best_model['name'], explanation)