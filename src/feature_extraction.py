# src/feature_extraction.py

import pyshark

def extract_features(packet):
    features = []
    try:
        # Extract packet length
        features.append(len(packet))  # Packet length
        # Extract transport layer
        features.append(packet.transport_layer)  # TCP/UDP
        # You can add other features here based on your dataset and requirements
    except:
        pass
    return features
