import cv2
import numpy as np


net = cv2.dnn.readNetFromDarknet("./yolo-coco/yolov3.cfg","./yolo-coco/yolov3.weights")
classes = []
with open("./yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


img = cv2.imread("./image/Screenshot3.png")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]

        label = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
        # cv2.putText(img, label, (x, y -5), font, 0.5, (255,255,255), 1)

img = cv2.resize(img, None, fx=1, fy=1)
cv2.imshow("Image", img)
cv2.imwrite('./image/new_image.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()