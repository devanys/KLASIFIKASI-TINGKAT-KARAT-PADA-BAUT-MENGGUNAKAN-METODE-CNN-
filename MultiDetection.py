from keras.models import load_model
import cv2
import numpy as np

model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
input_image_path = "Kumpulan.jpg"
image = cv2.imread(input_image_path)
orig_image = image.copy()
image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
image_for_prediction = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
image_for_prediction = (image_for_prediction / 127.5) - 1
prediction = model.predict(image_for_prediction)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]
print("Predicted Class:", class_name)
print("Confidence Score:", round(confidence_score * 100, 2), "%")
if "bad" in class_name.lower():
    bbox_color = (0, 0, 255)
else:
    bbox_color = (0, 255, 0)
gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    x = int(x * (image.shape[1] / image_resized.shape[1])) 
    y = int(y * (image.shape[0] / image_resized.shape[0]))  
    w = int(w * (image.shape[1] / image_resized.shape[1]))  
    h = int(h * (image.shape[0] / image_resized.shape[0]))  
    object_roi = orig_image[y:y+h, x:x+w]
    image_for_prediction = cv2.resize(object_roi, (224, 224), interpolation=cv2.INTER_AREA)
    image_for_prediction = np.asarray(image_for_prediction, dtype=np.float32).reshape(1, 224, 224, 3)
    image_for_prediction = (image_for_prediction / 127.5) - 1
    prediction = model.predict(image_for_prediction)
    class_index = np.argmax(prediction)
    class_name = class_names[class_index].strip() 
    confidence_score = prediction[0][class_index]
    if "bad" in class_name.lower():
        bbox_color = (0, 0, 255) 
    else:
        bbox_color = (0, 255, 0)
    cv2.rectangle(orig_image, (x, y), (x + w, y + h), bbox_color, 2)
    cv2.putText(orig_image, f"{class_name}: {round(confidence_score * 100, 2)}%",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)
cv2.imshow("UAS Kelompok 3 AMV", orig_image)
cv2.waitKey(0)
cv2.destroyAllWindows()