import os
import sys
from ultralytics import YOLOv10, YOLO
import torchvision
import torch
from PIL import Image
import numpy as np
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image
import cv2
from YOLOv10_Explainer import yolov10_heatmap

MODEL_DET_PATH = "models/detect/train_V10M/YOLOv10M.pt"
MODEL_CLS_PATH = "models/classification/train_cls/YOLOv8M-cls.pt"
TEST_PATH = "test/"

def detection_cam(img, conf_threshold=0.4, method="GradCAM", layer=[16]):  
    model = yolov10_heatmap(
        weight = MODEL_DET_PATH, 
        conf_threshold = conf_threshold, 
        device = 'cpu',
        method = method, 
        layer = layer,
        show_box = False,
    )
    imagelist = model(img)
    return imagelist[0]

def eigencam_cls(model_cls, img):
    cam = EigenCAM(model_cls, target_layers =[model_cls.model.model[-1].conv], task='cls')
    
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    img = cv2.resize(open_cv_image, (640, 640))
    rgb_img = img.copy()
    img = np.float32(img) / 255
    grayscale_cam = cam(rgb_img)[0, :, :]

    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    result_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
    return result_image
        
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0, (0, 0, 0, 0)
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    overlap_box = (inter_x_min, inter_y_min, inter_x_max, inter_y_max)
    
    return iou, overlap_box

def obscure_non_intersection(image, specific_box, boxes, box2):
    
    _, _, _, y_max = box2

    for box in boxes:
        x1_min, y1_min, x1_max, y1_max = box
        
        distance_x = x1_min.item() - specific_box[0].item()
        distance_y = y1_min.item() - specific_box[1].item()
        if distance_x < 0:
            distance_x = - distance_x
        if distance_y < 0:
            distance_y = - distance_y
        
        if  distance_x < int(y_max.item()*2) and distance_y < int(y_max.item()*2):
            iou, overlap_box = calculate_iou(specific_box, box)
            
            if iou > 0:
                inter_x_min, inter_y_min, inter_x_max, inter_y_max = map(int, overlap_box)                
                if y1_min < inter_y_min:
                    image[int(y1_min):int(inter_y_min), int(x1_min):int(x1_max)] = 0
                if inter_y_max < y1_max:
                    image[int(inter_y_max):int(y1_max), int(x1_min):int(x1_max)] = 0
                if x1_min < inter_x_min:
                    image[int(inter_y_min):int(inter_y_max), int(x1_min):int(inter_x_min)] = 0
                if inter_x_max < x1_max:
                    image[int(inter_y_min):int(inter_y_max), int(inter_x_max):int(x1_max)] = 0
            else:
                image[int(y1_min):int(y1_max), int(x1_min):int(x1_max)] = 0
        
    result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_image = torch.tensor(result_image, dtype=torch.float32)
    tensor = result_image / 255.0
    result_image = tensor.permute(2, 0, 1)
    
    return result_image

def main(image, cam_detection = False, cam_cls = False):
    
    model_det = YOLOv10(MODEL_DET_PATH)
    model_cls = YOLO(MODEL_CLS_PATH)

    result_path = f"Risultati/{os.path.splitext(os.path.basename(image))[0]}"
    classification_path = f"{result_path}/classificazione"

    os.makedirs(result_path, exist_ok=True)

    result = model_det(image, imgsz=512)
    
    if cam_detection:
        detection_cam(image).save(f'{result_path}/cam_detection.jpg')

    image_pil = Image.open(image)
    bounding_boxes = result[0].boxes.xyxy
    np_array_image2 = cv2.imread(image)

    for i, (box, box2) in enumerate(zip(result[0].boxes.xyxy, result[0].boxes.xywh)):
        
        x_min, y_min, x_max, y_max = box
        _, _, x_max1, y_max1 = box2
        
        top = int(y_min.item()-y_max1.item())
        left = int(x_min.item()-y_max1.item()-x_max1.item()/2)
        height = int(y_max1.item()*3)
        width = int(y_max1.item()*3)
        
        np_array_image = cv2.imread(image)
        result_image = obscure_non_intersection(np_array_image, box, bounding_boxes, box2)
        
        img_width, img_height = image_pil.size
        
        if top < 0:
            top = 0
        if left < 0:
            left = 0
        
        if  top + height > img_height:
            height = img_height - top
        if left + width > img_width:
            width = img_width - left
        
        img2 = torchvision.transforms.functional.crop(result_image, top, left, height, width)

        img3 = torchvision.transforms.ToPILImage()(img2) 
        
        result2 = model_cls(img3, imgsz=512)
        
        if cam_cls:
            os.makedirs(classification_path, exist_ok=True)
            img_res = eigencam_cls(model_cls, img3)
            cv2.imwrite(f'{classification_path}/eigen_cam{i}.jpg', img_res)
        
        label = int(torch.argmax(result2[0].probs.data).item())
        
        cv2.rectangle(np_array_image2, (int(x_min.item()), int(y_min.item())), (int(x_max.item()), int(y_max.item())), color=(0, 255, 0) if label == 1 else (255, 0, 0), thickness=2)
        cv2.putText(np_array_image2, f"{result2[0].names[label]} {round(result2[0].probs.data[label].item(), 2)}", (int(x_min.item()), int(y_min.item())-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if label == 1 else (255, 0, 0), 2)
        
    cv2.imwrite(f'{result_path}/detection.jpg', np_array_image2)
    cv2.destroyAllWindows()

if len(sys.argv) > 1:
    img_input = sys.argv[1]
    cam_det = eval(sys.argv[2]) if len(sys.argv) > 2 else False
    cam_cls = eval(sys.argv[3]) if len(sys.argv) > 3 else False
    if os.path.isdir(img_input):
        list_img = os.listdir(img_input)
        for img in list_img:
            main(f"{img_input}/{img}", cam_detection=cam_det, cam_cls=cam_cls)
    elif os.path.isfile(img_input):
        main(img_input, cam_detection=cam_det, cam_cls=cam_cls)
    else: print("Directory o immagine non validi!")
else:
    list_img = os.listdir(TEST_PATH)
    for img in list_img:
        main(TEST_PATH+img, cam_detection=False, cam_cls=False)
