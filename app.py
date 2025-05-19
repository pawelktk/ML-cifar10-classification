from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64
from src.models import create_simple_cnn_model, create_dropout_cnn_model, create_batchnorm_cnn_model, create_deep_cnn_model, create_mobilenet_transfer_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ładowanie wytrenowanych modeli
MODELS = {
    "Simple CNN": create_simple_cnn_model((32, 32, 3), 10),
    "Dropout CNN": create_dropout_cnn_model((32, 32, 3), 10),
    "BatchNorm CNN": create_batchnorm_cnn_model((32, 32, 3), 10),
    "Deep CNN": create_deep_cnn_model((32, 32, 3), 10),
    "MobileNet Transfer": create_mobilenet_transfer_model((32, 32, 3), 10)
}

# Wczytanie wag dla każdego modelu
for model_name in MODELS:
    model_path = f"models/{model_name.lower().replace(' ', '_')}.h5"
    if os.path.exists(model_path):
        MODELS[model_name].load_weights(model_path)

# Etykiety klas CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

def get_top_predictions(predictions, top_k=3):
    top_indices = np.argsort(predictions[0])[::-1][:top_k]
    return [(CLASS_NAMES[i], float(predictions[0][i]) * 100) for i in top_indices]

def generate_grad_cam(model, img_array, layer_name=None):
    if layer_name is None:
        # Domyślne warstwy dla różnych architektur
        if "MobileNet" in model.name:
            layer_name = "out_relu"
        else:
            layer_name = "conv2d_2" if "conv2d_2" in [l.name for l in model.layers] else model.layers[-3].name
    
    # Konwersja modelu na model Grad-CAM
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # Konwersja heatmapy na obraz
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((32, 32))
    heatmap = np.array(heatmap)
    
    return heatmap

def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

@app.route('/')
def index():
    return render_template('classifier.html', models=list(MODELS.keys()))

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(upload_path)
        
        selected_models = request.form.getlist('model')
        compare_mode = len(selected_models) > 1
        
        results = []
        heatmaps = []
        
        for model_name in selected_models:
            model = MODELS.get(model_name)
            if model is None:
                continue
                
            img_array = preprocess_image(upload_path)
            predictions = model.predict(img_array)
            top_preds = get_top_predictions(predictions)
            
            # Generowanie Grad-CAM
            try:
                heatmap = generate_grad_cam(model, img_array)
                plt.figure(figsize=(5,5))
                plt.imshow(heatmap, cmap='jet', alpha=0.5)
                plt.axis('off')
                heatmap_base64 = plot_to_base64()
                plt.close()
                heatmaps.append(heatmap_base64)
            except Exception as e:
                print(f"Error generating Grad-CAM: {e}")
                heatmaps.append(None)
            
            results.append({
                'name': model_name,
                'top_predictions': top_preds,
                'prediction': top_preds[0][0],
                'confidence': round(top_preds[0][1], 2)
            })
        
        return render_template('classifier.html',
                             models=list(MODELS.keys()),
                             selected_models=selected_models,
                             image_path=upload_path,
                             results=results,
                             heatmaps=heatmaps,
                             compare_mode=compare_mode,zip=zip)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)