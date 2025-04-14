import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import numpy as np
from numpy import random
import pathlib
import sys
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import time

# Add Yolo v7 Repository to Python path

pth_yolov7 = pathlib.Path(r'/home/roboticscorner/Python/yolov7')

if not str(pth_yolov7) in sys.path:
    sys.path.append(str(pth_yolov7))

from utils.general import non_max_suppression

input_video_path = "/home/roboticscorner/Python/people_with_helmet_working_1920_1080_50fps.mp4"

weights_path = "/home/roboticscorner/Downloads/best.pt"

Yolo_Detect_file = "/home/roboticscorner/Python/yolov7/detect.py"

# Initialize Torch device, model and get stride

device = torch.device('cpu')

model_path = weights_path
ckpt = torch.load(model_path, map_location=device)
model = ckpt['model'].float().fuse().eval()
for m in model.modules():
    if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
        m.inplace = True  # pytorch 1.7.0 compatibility
    elif type(m) is nn.Upsample:
        m.recompute_scale_factor = None  # torch 1.11.0 compatibility
model.half();

stride = int(model.stride.max().item())

def letterbox(im, new_width, stride):
    """Resizes image to new width while maintaining aspect ratio, and trims to ensure height is a multiple of stride."""
    new_width = int(new_width)
    h, w = im.shape[:2]
    r = new_width / w
    scaled_height = int(r * h)
    im = cv2.resize(im, (new_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    trim_rows = scaled_height % stride
    if trim_rows != 0:
        final_height = scaled_height - trim_rows
        offset = trim_rows // 2
        im = im[offset:(offset + final_height)]
    return im

def run_model(model, img, device):
    """Runs a PyTorch model on the input image tensor after preprocessing it."""
    img = np.expand_dims(img, 0)
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).half()
    img /= 255.0

    with torch.no_grad():
        return model(img)[0]
    
def plot_one_box(x, img, label, color):
    """Draws a rectangle on the input image, adds a label with the given color, and writes it on the image."""
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=1, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(label, 0, fontScale=1/3, thickness=1)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, 1/3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

def plot_boxes(img, pred, names, colors):
    """Draws rectangles and writes labels on an input image for each detection prediction from a list."""
    for det in pred:
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img, label, colors[int(cls)])

# Set the input image size and enable benchmark mode for CuDNN to speed up inference.
imgsz = 640
cudnn.benchmark = True

# Get the class names for the model and generate random colors for drawing boxes on the image.
names = model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
# Open the default camera for capturing video.
cap = cv2.VideoCapture('/home/roboticscorner/Python/people_with_helmet_working_1920_1080_50fps.mp4')

# Loop until the camera is closed.
try:
    while cap.isOpened():
        # Read a frame from the camera and ensure successfully read
        ret, im0 = cap.read()
        assert ret, "Failed to read"

        # Resize and pad the image to the specified size while maintaining the aspect ratio.
        img = letterbox(im0, imgsz, stride)
        
        # Run the model on the preprocessed image.
        pred = run_model(model, img, device)
        
        # Perform non-maximum suppression to remove overlapping boxes.
        pred = non_max_suppression(pred)
        
        # Draw the boxes on the image and display it.
        plot_boxes(img, pred, names, colors)
        cv2.imshow("YOLO v7 Demo", img)
        # Show frame
        # frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # clear_output(wait=True)
        # plt.imshow(frame_rgb)
        # plt.axis('off')
        # display(plt.gcf())

        #time.sleep(0.005)  # Optional delay
        
        # Exit the loop if the user presses the 'q' key.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the camera and close all windows.
    cap.release()
    cv2.destroyAllWindows()