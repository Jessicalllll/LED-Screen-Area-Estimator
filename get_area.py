import sys
import os
import cv2
import math
import numpy as np
from ultralytics import YOLO

# Define a dictionary for average areas of familiar objects from COCO dataset (in square meters)
coco_avg_areas = {
    0: 0.79,  # person
    1: 1.08,  # bicycle
    2: 8.1,  # car
    3: 0.3,  # traffic light
    4: 0.25,  # fire hydrant
    5: 0.71,  # stop sign
    6: 0.15,  # parking meter
    7: 0.75,  # bench
    8: 0.04,  # bird
    9: 0.785,  # umbrella
    10: 0.0236,  # bottle
    11: 0.0157,  # wine glass
    12: 0.00785,  # coffee cup
    13: 0.004,  # fork
    14: 0.015,  # kitchen knife
    15: 0.006,  # spoon
    16: 0.0177,  # bowl
    17: 0.5,  # chair
    18: 2,  # couch
    19: 0.25,  # flowerpot
    20: 1.5,  # kitchen & dining room table
    21: 0.06,  # laptop
    22: 0.0105,  # mobile phone
    23: 1.26,  # refrigerator
    24: 0.071,  # clock
    25: 0.053,  # vase
    26: 1.8,  # window
    27: 1.6,  # door
    28: 0.12,  # serving tray
    29: 0.071,  # plate
}

# Function to estimate LED screen area using detected objects
def estimate_led_area(led_bbox, detected_objects, image_height, image_width):
    closest_object = None
    min_distance = float('inf')

    led_x_center, led_y_center = led_bbox[0], led_bbox[1]

    for obj in detected_objects:
        label = obj['label']
        x_center, y_center, w, h = obj['bbox']
        obj_x_center, obj_y_center = x_center * image_width, y_center * image_height

        if label in coco_avg_areas:
            distance = np.sqrt((led_x_center - obj_x_center) ** 2 + (led_y_center - obj_y_center) ** 2)  # Euclidean distance
            if distance < min_distance:
                min_distance = distance
                closest_object = obj

    if closest_object is None:
        return None

    closest_label = closest_object['label']
    closest_pixel_area = closest_object['bbox'][2] * closest_object['bbox'][3]  # Pixel area of the closest common object
    real_area = coco_avg_areas[closest_label]  # Real area of the closest common object

    pixel_per_meter = closest_pixel_area / real_area

    led_pixel_area = led_bbox[2] * led_bbox[3] * image_width * image_height
    estimated_led_area = led_pixel_area / pixel_per_meter

    return estimated_led_area

# Function to calculate LED area using default values if no objects are detected
def calculate_led_area(position, temp_size, category, in_out, detected_objects, image_height, image_width):
    # Assuming position is a tuple (x1, y1, x2, y2)
    x1, y1, x2, y2 = position
    # Assuming temp_size is a tuple (width, height)
    img_width, img_height = temp_size
    
    # Calculate pixel area of LED screen
    led_width = abs(x2 - x1)
    led_height = abs(y2 - y1)
    pixel_area = led_width * led_height
    
    # Define default values for wall area based on category and indoor/outdoor
    default_led_area = {
        'indoor': {
            'Bar': 30.0,
            'Beverage': 20.0,
            'Cantonese': 25.0,
            'HairSalon': 15.0,
            'Hotpot': 28.0,
            'Japanese': 22.0,
            'Store': 35.0,
            'Szechuan': 27.0
        },
        'outdoor': {
            'Bar': 40.0,
            'Beverage': 30.0,
            'Cantonese': 35.0,
            'HairSalon': 25.0,
            'Hotpot': 38.0,
            'Japanese': 32.0,
            'Store': 45.0,
            'Szechuan': 37.0
        }
    }
    
    # Define the LED screen bounding box
    led_bbox = ((x1 + x2) / 2 / img_width, (y1 + y2) / 2 / img_height, (x2 - x1) / img_width, (y2 - y1) / img_height)

    # Estimate LED screen area using detected objects
    estimated_area = estimate_led_area(led_bbox, detected_objects, image_height, image_width)
    
    if estimated_area is not None:
        return round(estimated_area, 2)
    else:
        # Use default wall area to calculate the LED screen area
        led_area = default_led_area[in_out][category]
        return round(led_area, 2)

def main(image_path, position, temp_size, category, in_out, weights_path):
    # Load the custom trained YOLOv8 model
    model = YOLO(weights_path)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Run detection for common objects
    results = model(image_path)

    detected_objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            label = int(box.cls.item())  # Get the class label as an integer
            x_center, y_center, w, h = box.xywh[0].tolist()  # Convert tensor to list and unpack

            detected_objects.append({
                'label': label,
                'bbox': (x_center, y_center, w, h)
            })

    # Calculate LED screen area
    calculated_led_area = calculate_led_area(position, temp_size, category, in_out, detected_objects)

    print(f"Calculated LED area: {calculated_led_area} sq meters")

    return calculated_led_area
if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 get_area.py image_path position image_size category indoor_outdoor weights_path")
        sys.exit(1)

    # Extract command-line arguments
    d_image_path = sys.argv[1]
    d_position = tuple(map(int, sys.argv[2].split(',')))
    d_image_size = tuple(map(int, sys.argv[3].split(',')))
    d_category = sys.argv[4]
    d_indoor_outdoor = sys.argv[5]
    d_weights_path = sys.argv[6]

    result = main(d_image_path, d_position, d_image_size, d_category, d_indoor_outdoor, d_weights_path)
    print(result)
