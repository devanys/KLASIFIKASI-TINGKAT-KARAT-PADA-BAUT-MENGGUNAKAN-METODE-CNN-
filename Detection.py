from keras.models import load_model
import cv2
import numpy as np
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
input_image_path = "Kumpulan.jpg"
image = cv2.imread(input_image_path)
orig_image = image.copy()
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

# Make the image a numpy array and reshape it to the model's input shape
image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

# Normalize the image array
image = (image / 127.5) - 1

# Predict the model
prediction = model.predict(image)
index = np.argmax(prediction)
class_name = class_names[index].strip()  # Remove any extra whitespace

# Print prediction and confidence score
confidence_score = prediction[0][index]
print("Predicted Class:", class_name)
print("Confidence Score:", round(confidence_score * 100, 2), "%")

# Threshold for confidence score (adjust as needed)
threshold = 0.5

# If confidence score is above threshold, display bounding box
if confidence_score > threshold:
    # Define bounding box color (BGR)
    bbox_color = (0, 255, 0)  # Green color

    # Display bounding box on original image
    cv2.putText(orig_image, f"{class_name}: {round(confidence_score * 100, 2)}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Object Detection", orig_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Object detection confidence below threshold.")
