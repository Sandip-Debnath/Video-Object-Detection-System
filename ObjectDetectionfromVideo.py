import cv2

config_file_path = 'Dataset/ssd.pbtxt'
frozen_model_path = 'Dataset/frozen_inference_graph.pb'
detection_model = cv2.dnn_DetectionModel(frozen_model_path, config_file_path)

class_labels = []
labels_file_path = 'Dataset/labels.txt'

with open(labels_file_path, 'rt') as labels_file:
    class_labels = labels_file.read().rstrip('\n').split('\n')

print(class_labels)
print(len(class_labels))

detection_model.setInputSize(320, 320)
detection_model.setInputScale(1.0 / 127.5)
detection_model.setInputMean((127.5, 127, 5, 127.5))
detection_model.setInputSwapRB(True)

video_source = 'Dataset/video2.mp4'
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Video Format Not Supported')

font_scale = 4
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    class_indices, confidences, bounding_boxes = detection_model.detect(frame, confThreshold=0.55)
    print(class_indices)
    if len(class_indices) != 0:
        for class_index, confidence, boxes in zip(class_indices.flatten(), confidences.flatten(), bounding_boxes):
            if class_index <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 3)
                cv2.putText(frame, class_labels[class_index - 1], (boxes[0] + 10, boxes[1] + 40), font, font_scale,
                            color=(0, 255, 0), thickness=3)
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
