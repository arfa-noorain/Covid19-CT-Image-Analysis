# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 22:03:15 2021

@author: AFRA SHAIKH
"""

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')


@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/details.html')
def details():
   return render_template('details.html')


@app.route('/upload_ct.html')
def upload_ct():
   return render_template('upload_ct.html')



@app.route('/uploaded_ct', methods = ['POST', 'GET'])
def uploaded_ct():
   if request.method == 'POST':
        
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_ct.jpg'))

 
   vgg_ct = load_model('models/vgg_batch29.h5')

   image = cv2.imread('./flask app/assets/images/upload_ct.jpg')
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
   image = cv2.resize(image,(224,224))
   image = np.array(image) / 255
   image = np.expand_dims(image, axis=0)
   
   

   vgg_pred = vgg_ct.predict(image)
   probability = vgg_pred[0]
   print("VGG Predictions:")
   if probability[0] > 0.5:
      vgg_ct_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      vgg_ct_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(vgg_ct_pred)


   return render_template('results_ct.html',vgg_ct_pred=vgg_ct_pred)

if __name__ == '__main__':
   app.secret_key = ".."
   app.run()