import requests
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os

# Inicialización de Flask
app = Flask(__name__)

# Definición de las Clases y Carga del Modelo
names = ['Alfredo_Daza','Brando_Armas', 'Edwin_Osorio', 'Emanuel_Rojas','Jael_Estefanero']

# URL del modelo almacenado en AWS S3
MODEL_URL = 'https://your-bucket-name.s3.amazonaws.com/modelo_final_CNN.h5'

# Función para descargar el modelo
def download_model(model_url):
    response = requests.get(model_url)
    with open('modelo_final_CNN.h5', 'wb') as f:
        f.write(response.content)

# Descargar el modelo si no existe
if not os.path.exists('modelo_final_CNN.h5'):
    print('Descargando el modelo...')
    download_model(MODEL_URL)
    print('Modelo descargado exitosamente.')

# Cargar el modelo
model = load_model('modelo_final_CNN.h5')
print('Modelo cargado exitosamente. Verificar http://127.0.0.1:5000/')

# Función de Predicción del Modelo
def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convertir de BGR a RGB
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    x = np.asarray(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        prediction = names[np.argmax(preds)]
        confidence = np.max(preds) * 100  # Obtener el porcentaje de precisión
        print(f'PREDICCIÓN: {prediction}, CONFIANZA: {confidence:.2f}%')
        return f'{prediction} ({confidence:.2f}%)'
    return None

if __name__ == '__main__':
    app.run(debug=False, threaded=False)


