#%%%
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib as plt
from matplotlib import pyplot as plt
 
# %%
IMG_PATH = '../../sample-images/cat.jpg'
!curl https://raw.githubusercontent.com/PracticalDL/Practical-Deep-Learning-Book/master/sample-images/cat.jpg --output cat.jpg
IMG_PATH = 'cat.jpg'
# %%
img = image.load_img(IMG_PATH, target_size=(224, 224))
plt.imshow(img)
plt.show()
# %%
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0) # Increase the number of dimensions by one and add a new axis at the position of the axis parameter
# %%

#%%
model = tf.keras.applications.resnet50.ResNet50()
# %%
def clasifier(img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_preprocessed)
    print(decode_predictions(prediction, top=3)[0])
    #return prediction

#%%
result = clasifier(IMG_PATH)
# %%
!curl https://raw.githubusercontent.com/PracticalDL/Practical-Deep-Learning-Book/master/sample-images/dog.jpg --output dog.jpg
IMG_PATH = 'dog.jpg'
img=image.load_img(IMG_PATH, target_size=(224, 224))
plt.imshow(img)
plt.show()
# %%
clasifier(IMG_PATH)
# %%
