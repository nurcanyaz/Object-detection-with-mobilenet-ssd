import numpy as np
import os
import cv2

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "train", "tvmonitor"]

colors = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt","MobileNetSSD_deploy.caffemodel")

files = os.listdir()
img_path_list = []

for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)

for i in img_path_list:
    image = cv2.imread(i)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)

    detections = net.forward()

    for j in np.arange(0, detections.shape[2]):
        confidance = detections[0, 0, j, 2]

        if confidance> 0.3:
            idx = int(detections[0, 0, j, 1])
            box = detections[0, 0, j, 3:7]*np.array([w, h, w, h])

            (startx, starty, endx, endy) = box.astype("int")
            label = "{} {}".format(classes[idx], confidance)

            cv2.rectangle(image, (startx, starty), (endx, endy), colors[idx], 2)
            y = starty - 16 if starty - 16 > 15 else starty + 16
            cv2.putText(image, label, (startx, starty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    cv2.imshow("ssd", image)
    if cv2.waitKey(1) & 0xFF == ord("q"): continue