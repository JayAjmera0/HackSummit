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
            if prediction == 'Diseased: Cercospora Lead Spot':
                info = 'Cercospora leaf spot is a fungal disease that causes purple or brown spots on the leaves of infected plants. The spots may be surrounded by a yellow area. As the disease progresses, the spots enlarge and merge. The leaves may turn yellow and drop. Severely infected plants may be stunted. The disease is most severe in warm, humid weather.'
                treat = 'Management of Cercospora leaf spot involves planting resistant maize varieties, practicing crop rotation, and ensuring good plant spacing for adequate air circulation. Fungicides may be used if needed, but cultural practices are essential for long-term control.'
            elif prediction == 'Diseased:Common Rust':
                info = 'Common rust is a fungal disease that causes reddish-brown pustules on the leaves of infected plants. The pustules are filled with powdery, orange-colored spores that are easily rubbed off. Severely infected leaves turn yellow and drop. The disease is most severe in warm, humid weather.'
                treat = ' Management of common rust involves planting resistant maize varieties, practicing crop rotation, and ensuring good plant spacing for adequate air circulation. Fungicides may be used if needed, but cultural practices are essential for long-term control.'
            elif prediction == 'Diseased:Northern Leaf Blight':
                info = 'Northern corn leaf blight is a fungal disease that causes cigar-shaped lesions on the leaves of infected plants. The lesions are grayish-green, may measure up to 6 inches long, and may run parallel to the leaf veins. Severely infected leaves turn brown and dry up. The disease is most severe in cool, humid weather.'
                treat = 'Management of northern corn leaf blight involves planting resistant maize varieties, practicing crop rotation, and ensuring good plant spacing for adequate air circulation. Fungicides may be used if needed, but cultural practices are essential for long-term control.'
            else:
                info = 'The crop has no visible disease.'
                treat = 'Check the soil moisture and nutrient content.'
            # Convert prediction to a string or any desired format
            result = f'The prediction is: {prediction}'
            
            # Encode the image to base64 for HTML rendering
            _, buffer = cv2.imencode('.jpg', image)
            uploaded_image = base64.b64encode(buffer).decode()

            return render_template('index.html', result=result,info = info,treat = treat, uploaded_image=f"data:image/jpeg;base64,{uploaded_image}")
        else:
            return render_template('index.html', result='No image uploaded.')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True ,port=5000)
