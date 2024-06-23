import math
import pandas as pd
import cv2
import os
from PIL import Image, ImageStat
import matplotlib.pyplot as plt

# script 1: filtra e rimuove immagini dalla cartella
def remove_img_from_excel():
    os.chdir("/Users/riccardoderitis/Developer/Progetti/Progetto CV/Dataset definitivo/fishnet/images")
    list_dir = os.listdir()
    img_list = []
    excel = pd.read_excel('/Users/riccardoderitis/Downloads/labels/foid_labels_bbox_v020.xlsx')

    print(len(list_dir))

    for img in excel["img_id"]:
        if img+".jpg" in list_dir:
            img_list.append(img+".jpg")

    print(len(img_list))

    for img in list_dir:
        if img not in img_list:
            os.remove(img)
        
    print(len(list_dir))

# script 2: annota le immagini
def annotation_image():
    excel_file = '/Users/riccardoderitis/Developer/Progetti/Progetto CV/fishnet.xlsx'
    df = pd.read_excel(excel_file)

    output_dir = '/Users/riccardoderitis/Developer/Progetti/Progetto CV/images_with_annotations'

    unique_images = df['img_id'].unique()

    for image_name in unique_images:
        
        annotations = df[df['img_id'] == image_name]
        
        image_path = f'/Users/riccardoderitis/Developer/Progetti/Progetto CV/images/{image_name}.jpg'
        image = cv2.imread(image_path)
        
        if image is not None:
            for index, row in annotations.iterrows():
                xmin, ymin, xmax, ymax = row['x_min'], row['y_min'], row['x_max'], row['y_max']
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            output_path = os.path.join(output_dir, image_name+".jpg")
            cv2.imwrite(output_path, image)
        else:
            print(f'Immagine {image_name} non trovata.')

    print("Annotazioni completate.")

# script 3: crea file di testo per ogni immagine nel formato di yolo
def from_excel_to_yolo_format():
    excel_file = '/Users/riccardoderitis/Downloads/labels/foid_labels_bbox_v020.xlsx'
    output_dir = '/Users/riccardoderitis/Developer/Progetti/Progetto CV/Dataset definitivo/fishnet/labels'
    images_dir = '/Users/riccardoderitis/Developer/Progetti/Progetto CV/Dataset definitivo/fishnet/images'
    image_ext = '.jpg'

    df = pd.read_excel(excel_file)

    for _, row in df.iterrows():
        image_name = row['img_id']
        xmin = row['x_min']
        ymin = row['y_min']
        xmax = row['x_max']
        ymax = row['y_max']
        
        image_path = os.path.join(images_dir, image_name + image_ext)
        
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
        
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        annotation = f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n"
        
        annotation_file = os.path.join(output_dir, image_name + '.txt')
        with open(annotation_file, 'a') as f:
            f.write(annotation)

# script 4: filtra immagini in base alla luminosità
def filter_image_from_brightness():
    
    os.chdir("/Users/riccardoderitis/Developer/Progetti/Progetto CV/Dataset definitivo/fishnet/images")
    # images_giorno = '/Users/riccardoderitis/Developer/Progetti/Progetto CV/Dataset definitivo/fishnet/images/94be2fd8-23f0-11e9-9f43-13333a9e9bec.jpg'
    # images_notte = '/Users/riccardoderitis/Developer/Progetti/Progetto CV/Dataset definitivo/fishnet/images/94be3a6e-23f0-11e9-9f5d-8b849c0de133.jpg'
    for img in os.listdir():
        im1 = Image.open(img)
        #im2 = Image.open(images_notte)
        stat1 = ImageStat.Stat(im1)
        #stat2 = ImageStat.Stat(im2)
        r1,g1,b1 = stat1.mean
        #r2,g2,b2 = stat2.mean
        avg_brightness = math.sqrt(0.241*(r1**2) + 0.691*(g1**2) + 0.068*(b1**2))
        if avg_brightness < 90:
            print(f"{img} valore: {avg_brightness}")
        #print(math.sqrt(0.241*(r2**2) + 0.691*(g2**2) + 0.068*(b2**2)))

# script 5: disegna il bounding box secondo il fomrato di yolo
def draw_yolo_bounding_box():
    
    image_path = '/Users/riccardoderitis/Developer/Progetti/Progetto CV/Dataset definitivo/fishnet/images/94c0e98a-23f0-11e9-a5f7-73e011938fc2.jpg'
    yolo_coords = (0.34, 0.302962962962963, 0.09833333333333333, 0.2740740740740741)

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Destruttura le coordinate YOLO
    x_center, y_center, box_width, box_height = yolo_coords

    # Converti le coordinate YOLO in coordinate dell'immagine
    x_center = int(x_center * width)
    y_center = int(y_center * height)
    box_width = int(box_width * width)
    box_height = int(box_height * height)

    # Calcola i punti del bounding box
    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    # Disegna il bounding box sull'immagine
    color = (0, 255, 0)  # Verde
    thickness = 2
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    plt.imshow(image)
    plt.show()

draw_yolo_bounding_box()
