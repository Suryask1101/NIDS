import warnings
import numpy as np
import joblib
import logging

# Load the trained model and scaler
model = joblib.load("../models/nids_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Set THRESHOLD manually to control output (change this value to test different outputs)
THRESHOLD = 1.5  # Example: If THRESHOLD = 1.05, it will print "DDoS Attack"

# Set up logging
logging.basicConfig(filename="detection_logs.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# Suppress sklearn warnings
warnings.simplefilter(action="ignore", category=UserWarning)

# Mapping numerical predictions to actual labels with definitions
class_labels = {
    0: "Benign - Normal network traffic with no malicious activity.",
    1: "DDoS - Distributed Denial of Service attack, overwhelming the target with excessive traffic.",
    2: "DoS - Denial of Service attack from a single source, overloading the network or system.",
    3: "BruteForce - Repeated login attempts to guess passwords or credentials.",
    4: "SSH Attack - A targeted brute-force attack on SSH services to gain unauthorized access."
}

# Custom threshold-based classification
def get_forced_prediction(threshold):
    """Force output based on the manually set threshold value."""
    if threshold <= 0.1:
        return 0  # Benign
    elif threshold <= 0.5:
        return 1  # DDoS
    elif threshold <= 1.0:
        return 2  # DoS
    elif threshold <= 1.5:
        return 3  # BruteForce
    else:
        return 4  # SSH Attack

# Function to simulate real-time traffic feature extraction
def get_real_time_features():
    expected_features = 78  # Ensure correct feature count
    feature_vector = np.random.rand(1, expected_features)  
    return feature_vector

# Main function for real-time detection
def detect_intrusion():
    features = get_real_time_features()
    
    # Ensure features are scaled
    features_scaled = scaler.transform(features)

    # Get model predictions
    prediction_probabilities = model.predict_proba(features_scaled)[0]
    predicted_class_index = get_forced_prediction(THRESHOLD)  # Override with manual threshold logic
    predicted_class = class_labels.get(predicted_class_index, "Unknown - Unrecognized attack type.")  

    # Log the detection
    logging.info(f"Prediction: {predicted_class}, Probabilities: {prediction_probabilities}")

    # Print all class labels and descriptions
    print("\n--- Intrusion Detection System ---")
    print("Class Labels and Definitions:")
    for idx, desc in class_labels.items():
        print(f"{idx}: {desc}")

    print("\nDetection Results:")
    print(f"Predicted Class (Manually Controlled): {predicted_class}")
    print(f"Prediction Probabilities: {prediction_probabilities}")

    # Alert system
    if predicted_class_index != 0:  # If it's not 'Benign'
        print(f"Intrusion Detection Alert: {predicted_class} - Threat Detected!")
    else:
        print("No Threat Detected")

if __name__ == "__main__":
    detect_intrusion()
