import cv2
import numpy as np

def create_bounding_boxes(image_path, model_config, model_weights, class_names, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Use YOLO to create bounding boxes on an image.

    :param image_path: Path to the input image.
    :param model_config: Path to YOLO configuration file.
    :param model_weights: Path to YOLO pre-trained weights.
    :param class_names: List of class names corresponding to YOLO model.
    :param confidence_threshold: Minimum confidence for detections.
    :param nms_threshold: Non-maximum suppression threshold.
    :return: Image with bounding boxes drawn.
    """
    # Load YOLO model
    net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Load image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward pass
    outputs = net.forward(output_layers)

    # Process detections
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Draw bounding boxes
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(class_names[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

if __name__ == "__main__":
    # Paths to YOLO configuration, weights, and class names
    model_config = "yolov3.cfg"
    model_weights = "yolov3.weights"
    # Define class names directly in the script
    class_names = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    # Path to the input image
    image_path = "dataset/Training/Croc1/1.jpg"

    # Generate bounding boxes
    output_image = create_bounding_boxes(image_path, model_config, model_weights, class_names)

    # Display the output image
    cv2.imshow("Image with Bounding Boxes", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()