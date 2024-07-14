# LED Screen Area Estimator

## Introduction
Welcome to the LED Screen Area Estimator project! This Python script, `get_area.py`, calculates the area of an LED screen based on the coordinates of two opposite corners (x1, y1, x2, y2) and the width and height of the whole image. The script utilizes a YOLOv8 object detection model to identify common objects in the image. If objects are detected, it estimates the LED screen area using the closest detected object. If no objects are detected, it falls back to using default wall area values based on the specified category and whether the location is indoor or outdoor.

## Prerequisites
Before running this script, ensure you have the following installed:
- Python 3.x
- PyTorch
- Ultralytics YOLOv8

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/LED-Screen-Area-Estimator.git
    ```

2. **Navigate to the project directory**:
    ```sh
    cd LED-Screen-Area-Estimator
    ```

3. **Install the required libraries**:
    ```sh
    pip install torch ultralytics
    ```

4. **Download the Model Weights**:
    - Download the trained model weights from [this link](https://link_to_your_model_weights/best.pt) and save it in the project directory.

## How to Run the Script

### Arguments

The script requires the following command-line arguments:

- **image_path**: Path to the input image file.
- **x1,y1,x2,y2**: Coordinates of two opposite corners of the LED screen in the format "x1,y1,x2,y2".
- **width,height**: Width and height of the whole image in pixels, provided as "width,height".
- **category**: Category of the location including 'Bar', 'Beverage', 'Cantonese', 'HairSalon', 'Hotpot', 'Japanese', 'Store', 'Szechuan'.
- **indoor_outdoor**: Either "indoor" or "outdoor".
- **weights_path**: Path to the YOLOv8 model weights file.

### Example Command

```sh
python3 get_area.py "path/to/image.jpg" "393,550,517,660" "1500,900" "Bar" "indoor" "best.pt"

