import cv2
import numpy as np
 


weightsPath =  "yolov3.weights"
cfgPath =  "yolov3.cfg"
coco_namePath = "coco_names.txt"

def loading_dependencies():

    net = cv2.dnn.readNet(weightsPath, cfgPath)

    classes = []

    with open(coco_namePath, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    return net, output_layers, classes