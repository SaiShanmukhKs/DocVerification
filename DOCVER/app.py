from flask import Flask, render_template, request
import shutil
import cv2
from ultralytics import YOLO
import easyocr
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import os
from qreader import QReader
import cv2
qreader = QReader()

new_model = load_model('logoClassifier.h5')
model = YOLO("best.pt")
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    os.remove(r"static\image\user_upload.jpg")
    # Create a unique folder for each user's upload
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    upload_path = os.path.join(upload_folder, 'user_upload.jpg')

    # Save the user's uploaded image
    file = request.files['image']
    file.save(upload_path)


   
    
    


    # Perform the image processing as in your original code
    folder_path = 'runs'
    try:
        shutil.rmtree(folder_path)
        print(f"The folder {folder_path} and its contents have been successfully deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")

    
    result = model.predict(upload_path, save=True, save_crop=True)
    
    if(os.path.exists(r"runs\detect\predict\crops\IT\user_upload.jpg")):
        reader = easyocr.Reader(lang_list=["en"])
        text = reader.readtext(r"runs\detect\predict\crops\IT\user_upload.jpg")
        recognized_text = text[0][1] if text else "Text not recognized"
    else:
        shutil.move(r"runs\detect\predict\user_upload.jpg",r"static\image" )
        return render_template('index.html', result_message="Invalid Document")

    
    if(os.path.exists(r"runs\detect\predict\crops\Logo\user_upload.jpg")):
        img = cv2.imread(r"runs\detect\predict\crops\Logo\user_upload.jpg")
        plt.imshow(img)
        

    

        resize = tf.image.resize(img, (256, 256))
        plt.imshow(resize.numpy().astype(int))
    

        

        predict = new_model.predict(np.expand_dims(resize / 255, 0))

        logoID = -1 if predict > 0.5 else 1

        if recognized_text == "INCOME TAX DEPARTMENT" and logoID == 1:
            result_message = "Valid Document"
        else:
            result_message = "Invalid Document"


        
        shutil.move(r"runs\detect\predict\user_upload.jpg",r"static\image" )
            
        if(os.path.exists(r"runs\detect\predict\crops\qr_code\user_upload.jpg")):
            image = cv2.cvtColor(cv2.imread(r"runs\detect\predict\crops\qr_code\user_upload.jpg"), cv2.COLOR_BGR2RGB)

            decoded_text1 = qreader.detect_and_decode(image=image)

            def convertTuple(tup):
                # initialize an empty string
                str = ''
                for item in tup:
                    str = str + item
                return str

            decodedS= convertTuple(decoded_text1)


        return render_template('index.html', result_message=result_message,qr_data=decodedS)
    else:
        shutil.move(r"runs\detect\predict\user_upload.jpg",r"static\image" )
        return render_template('index.html', result_message="Invalid Document")


if __name__ == '__main__':
    app.run(debug=True)
