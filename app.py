import os
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
import tensorflow as tf
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import matplotlib
# Set the backend to 'Agg' to avoid thread-related GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create debug directory
debug_dir = os.path.join('static', 'debug')
os.makedirs(debug_dir, exist_ok=True)

# Path to models
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Load models
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load the models - handle potential missing files
leaf_detection_model = None
try:
    # Try the original filename first
    leaf_detection_model_path = os.path.join(MODELS_PATH, 'leaf_detection_model.tflite')
    if not os.path.exists(leaf_detection_model_path):
        # Try the filename with (1)
        leaf_detection_model_path = os.path.join(MODELS_PATH, 'leaf_detection_model (1).tflite')
    
    leaf_detection_model = load_model(leaf_detection_model_path)
    print(f"Successfully loaded leaf detection model from {leaf_detection_model_path}")
except Exception as e:
    print(f"Warning: Leaf detection model could not be loaded: {str(e)}")
    print("Will skip leaf detection step")

banana_model = load_model(os.path.join(MODELS_PATH, 'bananaleaf.tflite'))
coffee_model = load_model(os.path.join(MODELS_PATH, 'finalcoffee_model.tflite'))

# Class names for each model
banana_classes = [
    "Boron Deficiency",
    "Calcium Deficiency",
    "Healthy",
    "Iron Deficiency",
    "Magnesium Deficiency",
    "Manganese Deficiency",
    "Nitrogen Deficiency",
    "Phosphorus Deficiency",
    "Potassium Deficiency"
]

coffee_classes = [
    "Boron Deficiency",
    "Calcium Deficiency",
    "Healthy",
    "Iron Deficiency",
    "Magnesium Deficiency",
    "Manganese Deficiency",
    "Nitrogen Deficiency",
    "Phosphorus Deficiency",
    "Potassium Deficiency"
]

# Remedies database
remedies = {
    "Nitrogen Deficiency": {
        "symptoms": "Yellowing of older leaves starting from the leaf tip and progressing along the midrib; stunted growth; smaller leaves; fewer flowers and fruits.",
        "remedies": [
            "Apply nitrogen-rich fertilizers like urea, ammonium sulfate, or ammonium nitrate.",
            "Use organic fertilizers like compost, well-rotted manure, or blood meal.",
            "Consider planting nitrogen-fixing cover crops like legumes.",
            "Apply a balanced fertilizer with higher nitrogen content."
        ],
        "fertilizers": [
            {"name": "Urea (46-0-0)", "application": "Apply 100-150g per adult plant, spread around the base avoiding direct contact with stem", "frequency": "Every 2-3 months"},
            {"name": "Ammonium Sulfate (21-0-0)", "application": "Apply 200-300g per adult plant", "frequency": "Every 2 months"},
            {"name": "Calcium Nitrate (15.5-0-0)", "application": "Apply 200-250g per adult plant", "frequency": "Every 2-3 months"}
        ]
    },
    "Phosphorus Deficiency": {
        "symptoms": "Dark green leaves with purple/reddish hue on underside; stunted growth; delayed maturity; thin stems; poor fruit or flower development.",
        "remedies": [
            "Apply phosphate fertilizers like triple superphosphate or rock phosphate.",
            "Use bone meal or fish bone meal as organic alternatives.",
            "Maintain soil pH between 6-7 to enhance phosphorus availability.",
            "Apply compost or well-rotted manure to improve soil structure and phosphorus content."
        ],
        "fertilizers": [
            {"name": "Triple Superphosphate (0-46-0)", "application": "Apply 50-100g per adult plant", "frequency": "Every 3-4 months"},
            {"name": "Bone Meal (3-15-0)", "application": "Apply 100-200g per adult plant, work into soil", "frequency": "Every 3 months"},
            {"name": "Rock Phosphate (0-30-0)", "application": "Apply 100-150g per adult plant", "frequency": "Every 4-6 months"}
        ]
    },
    "Potassium Deficiency": {
        "symptoms": "Yellowing or browning leaf edges; weak stems; reduced disease resistance; poor fruit quality and reduced size; leaf curling.",
        "remedies": [
            "Apply potassium-rich fertilizers like potassium sulfate or muriate of potash.",
            "Use wood ash as an organic source of potassium.",
            "Apply compost or well-decomposed manure.",
            "Mulch with banana peels or use banana peel tea as a natural potassium supplement."
        ],
        "fertilizers": [
            {"name": "Potassium Sulfate (0-0-50)", "application": "Apply 100-150g per adult plant", "frequency": "Every 2-3 months"},
            {"name": "Muriate of Potash (0-0-60)", "application": "Apply 80-120g per adult plant", "frequency": "Every 2-3 months"},
            {"name": "NPK 13-13-21", "application": "Apply 150-200g per adult plant", "frequency": "Every 2 months"}
        ]
    },
    "Healthy": {
        "symptoms": "No deficiency detected. The plant appears to be healthy with proper nutrient levels.",
        "remedies": [
            "Continue with current maintenance practices.",
            "Follow regular fertilization schedule for preventive care.",
            "Monitor irrigation and ensure proper watering.",
            "Continue pest monitoring and management."
        ],
        "fertilizers": [
            {"name": "Balanced NPK (14-14-14)", "application": "Apply 150-200g per adult plant", "frequency": "Every 3 months"},
            {"name": "Organic Compost", "application": "Apply 2-3kg around the base of plant", "frequency": "Every 6 months"},
            {"name": "Liquid Seaweed Extract", "application": "Dilute according to package and apply as foliar spray", "frequency": "Monthly"}
        ]
    },
    "Boron Deficiency": {
        "symptoms": "Thickened, brittle, or curled leaves; terminal bud dieback; stunted root growth; hollow stems or fruits; reduced flowering.",
        "remedies": [
            "Apply boron-containing fertilizers such as borax or sodium tetraborate.",
            "Use soil amendments with micronutrients including boron.",
            "Apply foliar spray containing boron for immediate correction.",
            "Maintain soil pH between 6.0-7.0 for optimal boron availability."
        ],
        "fertilizers": [
            {"name": "Borax (11% B)", "application": "Apply 5-10g per adult plant, diluted in water", "frequency": "Every 6 months"},
            {"name": "Solubor (20% B)", "application": "Foliar spray at 1-2g per liter of water", "frequency": "Every 2-3 months"},
            {"name": "Micronutrient Mix with Boron", "application": "Apply as directed on package", "frequency": "Every 4-6 months"}
        ]
    },
    "Calcium Deficiency": {
        "symptoms": "Distorted new leaves; hooked leaf tips; stunted root growth; blossom end rot in fruits; leaf necrosis at tips and margins.",
        "remedies": [
            "Apply calcium-rich amendments like agricultural lime or gypsum.",
            "Use calcium nitrate fertilizer for immediate availability.",
            "Ensure proper soil pH (6.0-7.0) for optimal calcium uptake.",
            "Maintain consistent soil moisture to facilitate calcium movement through the plant."
        ],
        "fertilizers": [
            {"name": "Agricultural Lime (CaCO3)", "application": "Apply 500g-1kg per square meter of soil", "frequency": "Once per year"},
            {"name": "Gypsum (CaSO4)", "application": "Apply 200-400g per square meter", "frequency": "Every 6-12 months"},
            {"name": "Calcium Nitrate (15.5-0-0 + 19% Ca)", "application": "Apply 150-200g per adult plant", "frequency": "Every 2-3 months"}
        ]
    },
    "Iron Deficiency": {
        "symptoms": "Interveinal chlorosis (yellowing between veins) in young leaves while veins remain green; stunted growth.",
        "remedies": [
            "Apply iron chelates (Fe-EDDHA, Fe-DTPA) for immediate correction.",
            "Use iron sulfate for soil application.",
            "Lower soil pH if alkaline (above 7.0) to improve iron availability.",
            "Apply organic matter to improve soil structure and nutrient retention."
        ],
        "fertilizers": [
            {"name": "Iron Chelate (Fe-EDDHA)", "application": "Apply 5-10g per adult plant", "frequency": "Every 2-3 months"},
            {"name": "Iron Sulfate (FeSO4)", "application": "Apply 10-20g per adult plant", "frequency": "Every 3-4 months"},
            {"name": "Foliar Iron Spray", "application": "Spray on foliage at 2-3g per liter of water", "frequency": "Every 2-4 weeks until symptoms improve"}
        ]
    },
    "Magnesium Deficiency": {
        "symptoms": "Interveinal chlorosis in older leaves starting from leaf margins; leaf edges may appear yellow, bronze, or reddish; leaf tips may curl.",
        "remedies": [
            "Apply magnesium sulfate (Epsom salt) to the soil or as foliar spray.",
            "Use dolomitic limestone to provide both magnesium and calcium.",
            "Apply balanced fertilizers that contain magnesium.",
            "Avoid excessive potassium fertilization which can reduce magnesium uptake."
        ],
        "fertilizers": [
            {"name": "Magnesium Sulfate (Epsom Salt)", "application": "Apply 20-30g per adult plant", "frequency": "Every 2-3 months"},
            {"name": "Dolomitic Limestone", "application": "Apply 500g-1kg per square meter", "frequency": "Once per year"},
            {"name": "Magnesium Nitrate", "application": "Foliar spray at 10g per liter of water", "frequency": "Every 3-4 weeks until symptoms improve"}
        ]
    },
    "Manganese Deficiency": {
        "symptoms": "Interveinal chlorosis similar to iron deficiency but with smaller green veins; may have small brown spots; stunted growth.",
        "remedies": [
            "Apply manganese sulfate to soil or as foliar spray.",
            "Use micronutrient mixes containing manganese.",
            "Maintain soil pH between 5.5-6.5 for optimal manganese availability.",
            "Avoid over-liming which can reduce manganese uptake."
        ],
        "fertilizers": [
            {"name": "Manganese Sulfate (MnSO4)", "application": "Soil application at 5-10g per adult plant", "frequency": "Every 3-4 months"},
            {"name": "Manganese Chelate", "application": "Foliar application at 1-2g per liter water", "frequency": "Every 2-3 weeks until symptoms improve"},
            {"name": "Micronutrient Mix with Manganese", "application": "Apply as directed on package", "frequency": "Every 3-6 months"}
        ]
    }
}

# Additional plant-specific recommendations
plant_specific = {
    "banana": {
        "Nitrogen Deficiency": "Banana plants are heavy nitrogen feeders. Consider applying additional nitrogen during the vegetative growth phase.",
        "Phosphorus Deficiency": "For banana plants, phosphorus is especially important during the flowering and fruiting stages.",
        "Potassium Deficiency": "Bananas require high amounts of potassium for proper bunch development and fruit filling.",
        "Boron Deficiency": "Boron is critical for banana plants during fruit development. Deficiency can cause deformed fruits and poor bunch development.",
        "Calcium Deficiency": "Calcium deficiency in banana plants can lead to poor root development and weaker pseudostems.",
        "Iron Deficiency": "Iron deficiency in banana plants often shows as interveinal chlorosis in young leaves, while older leaves remain green.",
        "Magnesium Deficiency": "Magnesium is important for chlorophyll production in banana plants. Deficiency often shows as 'Christmas tree' pattern on older leaves.",
        "Manganese Deficiency": "Manganese deficiency in banana plants can reduce disease resistance and affect overall plant vigor."
    },
    "coffee": {
        "Nitrogen Deficiency": "Coffee plants require consistent nitrogen supply throughout the year, with higher needs during flowering.",
        "Phosphorus Deficiency": "For coffee plants, adequate phosphorus is critical for root development and bean quality.",
        "Potassium Deficiency": "Coffee plants with potassium deficiency often show reduced resistance to coffee leaf rust and other diseases.",
        "Boron Deficiency": "Boron deficiency in coffee plants can affect flowering and fruit set, leading to reduced yield.",
        "Calcium Deficiency": "Calcium is essential for cell wall strength in coffee plants and helps with drought resistance.",
        "Iron Deficiency": "Iron deficiency in coffee shows as yellowing between leaf veins while the veins remain green, primarily in new growth.",
        "Magnesium Deficiency": "Coffee plants need adequate magnesium for chlorophyll production and carbohydrate metabolism.",
        "Manganese Deficiency": "Manganese is important for photosynthesis in coffee plants. Deficiency can cause interveinal chlorosis similar to iron deficiency."
    }
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path, target_size=(224, 224), add_batch_dim=True):
    """Preprocess image for model input to match original training preprocessing"""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    
    # Normalize using the same method as in the model training code
    img = img / 255.0
    
    if add_batch_dim:
        img = np.expand_dims(img, axis=0)
    
    print(f"Image preprocessed with range: {img.min():.5f} to {img.max():.5f}, mean: {img.mean():.5f}")
    return img.astype(np.float32)

def is_leaf(image_path):
    """Detect if the image contains a leaf using the leaf detection model"""
    # If no leaf detection model is available, assume all images contain leaves
    if leaf_detection_model is None:
        print("No leaf detection model available - assuming all images contain leaves")
        return True
    
    try:
        # Get input details to determine expected shape
        input_details = leaf_detection_model.get_input_details()
        input_shape = input_details[0]['shape']
        print(f"Leaf detection model expects input shape: {input_shape}")
        
        # Process image using the same method as in test code
        # Read image with OpenCV (BGR format)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Resize to the expected dimensions (224x224 by default)
        img_resized = cv2.resize(img, (224, 224))
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        print(f"Preprocessed image shape: {img_batch.shape}")
        print(f"Preprocessed image range: {img_batch.min():.5f} to {img_batch.max():.5f}, mean: {img_batch.mean():.5f}")
        
        # Ensure the input shape matches what the model expects
        if img_batch.shape != tuple(input_shape):
            print(f"Reshaping input from {img_batch.shape} to {tuple(input_shape)}")
            img_batch = np.reshape(img_batch, input_shape)
        
        # Set input tensor
        leaf_detection_model.set_tensor(input_details[0]['index'], img_batch)
        
        # Run inference
        leaf_detection_model.invoke()
        
        # Get output tensor
        output_details = leaf_detection_model.get_output_details()
        output_data = leaf_detection_model.get_tensor(output_details[0]['index'])
        
        # Print output shape for debugging
        print(f"Leaf detection model output shape: {output_data.shape}")
        print(f"Leaf detection model output: {output_data}")
        
        # Handle different output formats
        try:
            # If output is [not_leaf, leaf]
            if len(output_data.shape) > 1 and output_data.shape[-1] >= 2:
                probability_leaf = output_data[0][1]
            # If output is a single value (0=not leaf, 1=leaf)
            else:
                probability_leaf = float(output_data[0][0] if len(output_data.shape) > 1 else output_data[0])
            
            print(f"Leaf probability: {probability_leaf}")
            return probability_leaf > 0.5
        except Exception as e:
            print(f"Error processing leaf detection output: {str(e)}")
            # If there's any error, assume it is a leaf to continue with detection
            return True
            
    except Exception as e:
        print(f"Error during leaf detection inference: {str(e)}")
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")
        # If there's any error, assume it is a leaf to continue with detection
        return True

def get_model_info(model_name):
    """Get model-specific information"""
    model_info = {
        'bananaleaf.tflite': {
            'input_size': (224, 224),
            'preprocess': 'default',
            'description': 'Banana Leaf Model - Ensemble MobileNetV3Large & ResNet50V2'
        },
        'finalcoffee_model.tflite': {
            'input_size': (224, 224),
            'preprocess': 'default',
            'description': 'Coffee Leaf Model - Ensemble MobileNetV3Large & ResNet50V2 with Attention Layer'
        },
        'leaf_detection_model.tflite': {
            'input_size': (224, 224),
            'preprocess': 'default',
            'description': 'Leaf Detection Model'
        },
        'leaf_detection_model (1).tflite': {
            'input_size': (224, 224),
            'preprocess': 'default',
            'description': 'Leaf Detection Model'
        }
    }
    
    return model_info.get(model_name, {
        'input_size': (224, 224),
        'preprocess': 'default',
        'description': 'Unknown Model'
    })

def predict_deficiency(image_path, plant_type):
    """Predict nutrient deficiency based on plant type"""
    if plant_type == "banana":
        model = banana_model
        classes = banana_classes
        model_name = 'bananaleaf.tflite'
    else:  # coffee
        model = coffee_model
        classes = coffee_classes
        model_name = 'finalcoffee_model.tflite'
        
    # Get model-specific information
    model_info = get_model_info(model_name)
    print(f"Using {model_info['description']}")
    
    # Get input details to determine expected shape
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    input_shape = input_details[0]['shape']
    output_shape = output_details[0]['shape']
    print(f"{plant_type} model expects input shape: {input_shape}")
    print(f"{plant_type} model output shape: {output_shape}")
    
    # Ensure the classes match the model output
    expected_classes = output_shape[-1] if len(output_shape) > 1 else 1
    if expected_classes != len(classes):
        print(f"WARNING: Model outputs {expected_classes} classes but we have {len(classes)} class names defined")
    
    # Process image using same method as in the testing code
    try:
        # Read image with OpenCV (BGR format)
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Failed to read image at {image_path}")
            raise Exception("Image could not be read")
            
        # Save a debug copy of the original image
        debug_original = os.path.join('static', 'debug', f'original_{os.path.basename(image_path)}')
        cv2.imwrite(debug_original, img)
        print(f"Saved original image copy to {debug_original}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Resize to the expected dimensions
        target_size = model_info['input_size']
        img_resized = cv2.resize(img, target_size)
        
        # Save a debug copy of the resized image
        debug_resized = os.path.join('static', 'debug', f'resized_{os.path.basename(image_path)}')
        cv2.imwrite(debug_resized, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        print(f"Saved resized image to {debug_resized}")
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        print(f"Preprocessed image shape: {img_batch.shape}")
        print(f"Preprocessed image range: {img_batch.min():.5f} to {img_batch.max():.5f}, mean: {img_batch.mean():.5f}")
        
        # Ensure the input shape matches what the model expects
        if img_batch.shape != tuple(input_shape):
            print(f"Reshaping input from {img_batch.shape} to {tuple(input_shape)}")
            img_batch = np.reshape(img_batch, input_shape)
        
        # Set input tensor
        model.set_tensor(input_details[0]['index'], img_batch)
        
        # Run inference
        print(f"Running inference with {plant_type} model...")
        model.invoke()
        
        # Get output tensor
        output_data = model.get_tensor(output_details[0]['index'])
        
        # Print output shape and sample for debugging
        print(f"{plant_type} model output shape: {output_data.shape}")
        print(f"{plant_type} model output raw data: {output_data.flatten()}")
        
        # Get prediction - handle different output shapes
        if len(output_data.shape) > 1:
            prediction = output_data[0]  # First batch item
        else:
            prediction = output_data  # No batch dimension
        
        # Validate prediction array
        if len(prediction) < 1:
            print("ERROR: Empty prediction array")
            raise Exception("Model returned empty prediction")
            
        if len(prediction) != len(classes):
            print(f"WARNING: Prediction length ({len(prediction)}) doesn't match classes length ({len(classes)})")
            # If prediction has more elements than classes, truncate
            if len(prediction) > len(classes):
                print(f"Truncating prediction from {len(prediction)} to {len(classes)} elements")
                prediction = prediction[:len(classes)]
        
        print(f"Prediction array: {prediction}")
            
        class_idx = np.argmax(prediction)
        confidence = float(prediction[class_idx])
        
        print(f"Predicted class index: {class_idx}, confidence: {confidence:.4f}")
        if class_idx < len(classes):
            print(f"Class name: {classes[class_idx]}")
        else:
            print(f"ERROR: Class index {class_idx} is out of range for classes (length {len(classes)})")
            raise Exception(f"Invalid class index: {class_idx}")
        
        # Create a dictionary of class probabilities
        all_probabilities = {}
        for i in range(min(len(prediction), len(classes))):
            all_probabilities[classes[i]] = float(prediction[i])
        
        result = {
            "class": classes[class_idx],
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }
        
        return result
        
    except Exception as e:
        print(f"Error during {plant_type} leaf analysis: {str(e)}")
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")
        # Return a default result if there's an error
        return {
            "class": "Analysis Error",
            "confidence": 0,
            "all_probabilities": {class_name: 0 for class_name in classes}
        }

def save_inference_visualization(image_path, result, plant_type, save_dir='static/debug'):
    """Save a visualization of the inference for debugging"""
    try:
        # Create debug directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path} for visualization")
            return
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Input Image: {os.path.basename(image_path)}")
        plt.axis('off')
        
        # Plot probabilities - sort them for better visualization
        plt.subplot(1, 2, 2)
        classes = list(result['all_probabilities'].keys())
        probabilities = [result['all_probabilities'][cls] for cls in classes]
        
        # Sort by probability (descending)
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_probs = [probabilities[i] for i in sorted_indices]
        
        # Use colors to highlight the predicted class
        colors = ['green' if cls == result['class'] else 'gray' for cls in sorted_classes]
        
        y_pos = np.arange(len(sorted_classes))
        bars = plt.barh(y_pos, sorted_probs, color=colors)
        plt.yticks(y_pos, sorted_classes)
        plt.xlabel('Probability')
        plt.title(f'Deficiency Prediction ({plant_type.capitalize()})')
        
        # Add percentage labels to bars
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{sorted_probs[i]:.2%}',
                va='center'
            )
        
        # Add detection info
        detection_info = f"Detected: {result['class']}\nConfidence: {result['confidence']:.2%}"
        plt.figtext(0.5, 0.01, detection_info, wrap=True, horizontalalignment='center', fontsize=12)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plant_type}_inference_{timestamp}.jpg"
        
        # Ensure forward slashes for web paths
        web_safe_save_dir = save_dir.replace('\\', '/')
        debug_path = os.path.join(web_safe_save_dir, filename).replace('\\', '/')
        
        plt.tight_layout()
        plt.savefig(debug_path, dpi=100)
        plt.close()
        
        print(f"Debug visualization saved to {debug_path}")
        
        # Return only the part of the path after 'static/' to be used with url_for
        # This fixes the issue with static/static in URLs
        if debug_path.startswith('static/'):
            # For Flask's url_for, return just the path after static/
            return debug_path
        else:
            # If the save_dir doesn't start with 'static/', return the filename only
            # to be appended to 'debug/'
            return os.path.join('debug', filename).replace('\\', '/')
    except Exception as e:
        print(f"Error creating debug visualization: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('detect.html', error="No file selected")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('detect.html', error="No file selected")
        
        plant_type = request.form.get('plant_type', 'banana')
        if plant_type not in ['banana', 'coffee']:
            return render_template('detect.html', error="Invalid plant type")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Temporarily skipping leaf detection step
            # is_leaf_result = is_leaf(filepath)
            # if not is_leaf_result:
            #     return render_template(
            #         'result.html', 
            #         error="No leaf detected in the image. Please upload an image containing a clear leaf."
            #     )
            
            # Go directly to deficiency prediction
            result = predict_deficiency(filepath, plant_type)
            
            deficiency_class = result["class"]
            
            # Check if analysis encountered an error
            if deficiency_class == "Analysis Error":
                return render_template(
                    'result.html',
                    error="An error occurred analyzing the image. Please try a different image or try again later."
                )
                
            confidence = result["confidence"] * 100  # Convert to percentage
            all_probabilities = result["all_probabilities"]
            
            # Get remedies and recommendations
            deficiency_info = remedies.get(deficiency_class, {})
            plant_recommendation = plant_specific.get(plant_type, {}).get(deficiency_class, "")
            
            # Save inference visualization
            visualization_path = save_inference_visualization(filepath, result, plant_type)
            
            # Fix image path handling - ensure it's relative to static folder
            # The image is in static/uploads, but url_for('static') already points to the static directory
            # So we just need the path after 'static/'
            if filepath.startswith('static/') or filepath.startswith('static\\'):
                # Convert Windows backslashes to forward slashes for web URLs
                image_path = filepath.replace('\\', '/')
                # If it starts with static/, remove it for url_for
                if image_path.startswith('static/'):
                    image_path = image_path[7:]  # Remove 'static/'
            else:
                image_path = os.path.join('uploads', filename).replace('\\', '/')
            
            return render_template(
                'result.html',
                plant_type=plant_type,
                image_path=image_path,
                deficiency_class=deficiency_class,
                confidence=confidence,
                all_probabilities=all_probabilities,
                deficiency_info=deficiency_info,
                plant_recommendation=plant_recommendation,
                visualization_path=visualization_path
            )
            
        else:
            return render_template('detect.html', error="Invalid file type. Please upload .jpg, .jpeg, or .png files.")
    
    return render_template('detect.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True) 