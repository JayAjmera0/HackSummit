from flask import Flask, request, send_file
import os
from PIL import Image
from io import BytesIO
import cv2
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
 

def perform_ndvi_analysis(color_image_path, nir_image_path):
    
    # Read the images
    color_image = cv2.imread(color_image_path, cv2.IMREAD_COLOR)
    nir_image = cv2.imread(nir_image_path, cv2.IMREAD_COLOR)

    # Convert images to gray scale (assuming the red channel represents the color image and the NIR image is grayscale)
    red_channel = color_image[:, :, 0]
    nir_channel = nir_image[:, :, 0]

    # NDVI Calculation #
    # Assuming that the NIR and Red channels have been aligned and are of the same shape
    numerator = nir_channel.astype(float) - red_channel.astype(float)
    denominator = nir_channel.astype(float) + red_channel.astype(float)

    # Avoid division by zero
    denominator[denominator == 0] = 1e-9
    ndvi = numerator / denominator

    # Applying color map
    color_map = plt.cm.RdYlGn
    ndvi_colored = color_map(ndvi)
    output_image = (ndvi_colored * 255).astype(np.uint8)
    return output_image

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = './' 

 

@app.route('/', methods=['GET', 'POST'])

def index():

    if request.method == 'POST':

        # Fetch the uploaded files

        color_image = request.files.get('color_image')
        nir_image = request.files.get('nir_image')
        color_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'color_image.tif')

        nir_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'nir_image.tif')

        # Perform NDVI analysis
        output_image = perform_ndvi_analysis(color_image_path, nir_image_path)

        # Convert output image to send as a response
        output_image_pil = Image.fromarray(output_image)

        byte_io = BytesIO()
        output_image_pil.save(byte_io, 'PNG')

        byte_io.seek(0)

        return send_file(byte_io, mimetype='image/png')
    with open('./templates/ndvi.html', 'r') as file:

        form_html = file.read()
    return form_html
if __name__ == '__main__':
    app.run(debug=True)