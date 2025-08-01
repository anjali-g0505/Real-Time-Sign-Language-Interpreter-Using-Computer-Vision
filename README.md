# 🤟 Real-time Sign Language Interpreter
This project develops a real-time system to detect and interpret American Sign Language (ASL) gestures from a live video feed using computer vision and deep learning techniques. It leverages the TensorFlow Object Detection API for efficient and accurate gesture recognition.

# Features
Real-time Gesture Detection: Identifies predefined ASL signs from a live webcam feed.

Custom Dataset: Utilizes a custom-collected and annotated dataset of hand gestures.

Transfer Learning: Employs a pre-trained object detection model (SSD MobileNet V2) for faster and more accurate training.

Efficient Inference: Designed for relatively fast processing, suitable for real-time applications.

# Technologies Used
Python: The primary programming language.

OpenCV (cv2): For webcam access, image capture, and real-time video processing.

TensorFlow: The core deep learning framework.

TensorFlow Object Detection API: A powerful framework for building, training, and deploying object detection models.

LabelImg: A graphical image annotation tool used for drawing bounding boxes and labeling objects.

NumPy: For numerical operations, especially with image data.

Protocol Buffers (.pbtxt): Used for configuration and label mapping.

# Project Workflow
This project follows a standard object detection pipeline, with key stages focused on data preparation, model configuration, and real-time inference.

1. 📸 Data Collection
Role: Responsible for capturing raw image data.

Process: Used OpenCV to access the webcam and programmatically capture images of various sign language gestures (e.g., 'hello', 'yes', 'no', 'thanks', 'A'). Each gesture was captured multiple times to build a diverse initial dataset.

Output: Images stored in Tensorflow/workspace/collectedimages/images/{label}/.

2. 📝 Image Annotation
Role: Performed detailed image annotation.

Process: Employed LabelImg to manually draw precise bounding boxes around the hand gestures in each image and assign the corresponding sign label. This step creates XML files containing the object's location and class.

Significance: Provides the ground truth for the model to learn from, teaching it both what the object is and where it is located.

3. 📊 Data Splitting & TFRecord Generation
Role: Managed data splitting and conversion.

Process: Split the annotated image dataset into training and testing sets. Subsequently, converted these images and their annotations into TFRecords (train.record, test.record).

Significance: TFRecords are a highly optimized binary format for TensorFlow, enabling faster and more efficient data loading during model training.

4. 🧠 Model Selection & Configuration
Role: Researched and configured the base model.

Process: Researched various object detection architectures and selected SSD MobileNet V2 FPNLite from the TensorFlow Model Zoo. This choice was made due to its excellent balance of speed and accuracy, making it ideal for real-time inference on standard hardware.

Configuration: Configured the model's pipeline.config file, setting crucial hyperparameters (e.g., num_classes, batch_size) and enabling transfer learning by pointing to the pre-trained model's checkpoint. This allowed us to leverage pre-learned features and significantly reduce training time and data requirements.

Output: label_map.pbtxt (mapping class IDs to names) and a custom pipeline.config for training.

5. 🏋️ Model Training
Process: The model was trained using the TensorFlow Object Detection API's model_main_tf2.py script, leveraging the configured pipeline.config and the generated TFRecords. The training process involved fine-tuning the pre-trained SSD MobileNet model on our custom sign language dataset.

6. 🚀 Real-time Detection
Process: The trained model's checkpoint was loaded, and an optimized detection function was defined. This function processes live frames from the webcam, performs inference, and visualizes the detected signs with bounding boxes and confidence scores in real-time using OpenCV.

Set up TensorFlow Object Detection API: Follow the official TensorFlow Models repository instructions to set up the API and download pre-trained models.

Collect and Annotate Data: Run the image collection script and use LabelImg to annotate your custom signs.

Generate TFRecords: Use the provided scripts (often from the TF OD API) to convert annotations to TFRecords.

Configure Model: Update the pipeline.config with your specific dataset paths and hyperparameters.

Train the Model: Execute the training command using model_main_tf2.py.

Run Real-time Detection: Use the inference script to test the model on your webcam.
