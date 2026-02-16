import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("model/cats_vs_dogs_model.h5")

img = image.load_img("test2.jpg", target_size=(150,150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)

if prediction[0] > 0.5:
    print("🐶 Dog")
else:
    print("🐱 Cat")
