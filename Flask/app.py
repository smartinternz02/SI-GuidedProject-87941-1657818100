# Import Libraries:
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request

app = Flask(__name__)

# Loading Models:
model_1 = load_model('fruit.h5')
model_2 = load_model('vegitable.h5')

# Home Page:
@app.route('/')
def home():
    return render_template("home.html")

# Prediction Page:
@app.route('/prediction')
def prediction():
    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f=request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        
        img = image.load_img(filepath, target_size = (128,128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        plant = request.form['plant']
        print(plant)
        
        # Prediction for Fruits or Vegitables:
        if(plant == 'fruit'):
            # Prediction for Fruits:
            preds = np.argmax(model_1.predict(x), axis=1)
            #print(preds)
            df = pd.read_excel('precautions - fruits.xlsx')
            print(df.iloc[preds[0]]['caution'])
        else:
            # Prediction for Vegitables:
            preds = np.argmax(model_2.predict(x), axis=1)
            #print(preds)
            df = pd.read_excel('precautions - veg.xlsx')
            print(df.iloc[preds[0]]['caution'])
        return df.iloc[preds[0]]['caution']
        
    
if __name__ == '__main__':
    app.run(debug=False)