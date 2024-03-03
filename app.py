from flask import Flask, request, render_template
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('cifar_10_accuracy_8581.h5')

@app.route('/')
def home():
    return render_template('home.html', current_page='home')

@app.route('/about')
def about():
    return render_template('about.html', current_page='about')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    classes = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

    try:
        # using 'file' because it direct to the file name used in input tag
        uploaded_image = request.files['file']
        img = Image.open(uploaded_image)

        # checking if the image is rgb
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize the image to match model input size, converting to nup array, normalizing
        img = img.resize((32, 32))
        img_array = image.img_to_array(img)
        img_array /= 255.0

        # Add batch dimension which the model expects, so the shape becomes (1, height width, channels)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        return render_template('predict.html', predict_class=classes[int(predicted_class)].capitalize())
    except Exception as e:
        print(e)
        return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)
