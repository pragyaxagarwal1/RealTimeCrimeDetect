from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('best_model.h5')

# Function to make predictions on new images
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Resize image to match model input
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get the index of the highest probability
    return predicted_class

# Test on a new image
img_path = 'path_to_test_image.jpg'  # Replace with your image path
predicted_class = predict_image(img_path)

print(f'Predicted Class: {predicted_class}')
