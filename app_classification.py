import cv2
import numpy as np
from flask import Flask, request, render_template
import pickle
import base64

app = Flask(__name__)

def load_model():
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except Exception as e:
        return str(e)

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.files['imageInput']
        if input_data:
            # Read and decode the uploaded image
            image = cv2.imdecode(np.fromstring(input_data.read(), np.uint8), cv2.IMREAD_COLOR)

            # Resize the image
            image = cv2.resize(image, (100, 100))
            
            # Ensure that the image shape matches the model's input shape
            prediction = model.predict(np.expand_dims(image, axis=0))[0]
            classname = ['Diseased: Cercospora Lead Spot', 'Diseased:Common Rust', 'Diseased:Northern Leaf Blight', 'Healthy']
            prediction = classname[np.argmax(prediction)]
            
            # Convert prediction to a string or any desired format
            result = f'The prediction is: {prediction}'
            
            # Encode the image to base64 for HTML rendering
            _, buffer = cv2.imencode('.jpg', image)
            uploaded_image = base64.b64encode(buffer).decode()
            
            return render_template('index.html', result=result, uploaded_image=f"data:image/jpeg;base64,{uploaded_image}")
        else:
            return render_template('index.html', result='No image uploaded.')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True ,port=5000)
