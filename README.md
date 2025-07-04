# Plant Nutrient Deficiency Detection System

A web-based application for detecting nutrient deficiencies in banana and coffee plants using AI.

## Overview

Plant Nutrient Detective is an advanced application that leverages deep learning models to identify deficiencies in plant leaves. By analyzing uploaded images, the system provides instant results and detailed recommendations for treatment.

## Features

- **Leaf Detection**: Verifies that uploaded images contain plant leaves
- **Plant Type Selection**: Supports both banana and coffee plants
- **Deficiency Analysis**: Detects nitrogen, phosphorus, and potassium deficiencies
- **Detailed Results**: Provides confidence scores and probability distributions
- **Treatment Recommendations**: Offers specific remedies and fertilizer suggestions
- **User-Friendly Interface**: Modern, responsive design for all devices

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, Bootstrap 5, JavaScript
- **AI Models**: TensorFlow Lite models (MobileNetV3Large, ResNet50V2)
- **Image Processing**: OpenCV, Pillow

## Setup and Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/plant-nutrient-detective.git
   cd plant-nutrient-detective
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Copy your TFLite models to the `models` directory:

   - `models/leaf_detection_model.tflite` - Model for detecting leaves
   - `models/bananaleaf.tflite` - Model for banana leaf deficiency detection
   - `models/finalcoffee_model.tflite` - Model for coffee leaf deficiency detection

5. Run the application:

   ```
   python app.py
   ```

6. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Model Architecture

The application uses ensemble models combining MobileNetV3Large and ResNet50V2 architectures, optimized and converted to TensorFlow Lite format for efficient inference.

## Directory Structure

```
plant-nutrient-deficiency-detection/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── models/               # TFLite model files
│   ├── leaf_detection_model.tflite
│   ├── bananaleaf.tflite
│   └── finalcoffee_model.tflite
├── static/               # Static assets
│   ├── css/              # CSS files
│   ├── js/               # JavaScript files
│   ├── images/           # Image assets
│   └── uploads/          # User uploaded images
└── templates/            # HTML templates
    ├── index.html        # Home page
    ├── detect.html       # Upload and detection page
    ├── result.html       # Results display page
    └── about.html        # About the project page
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Coffee and Banana leaf deficiency dataset contributors
- The TensorFlow team for TFLite
