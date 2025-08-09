"""
Image classifier using ResNet 

"""

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet') 

img_path = ''  # TODO: figure out how to link this with the frontend

img = image.load_img(img_path, target_size=(224, 224))  # resize
img_array = image.img_to_array(img) 
img_array = np.expand_dims(img_array, axis=0)  # batch dimension (required by the model)
img_array = preprocess_input(img_array) 

preds = model.predict(img_array)  # Predict the class of the image
print('Predicted:', decode_predictions(preds, top=3)[0])  # Decode the predictions into readable class labels
