"""
CardioInsight AI - Multimodal Analysis Example

This example demonstrates how to use the CardioInsight AI system for multimodal ECG analysis,
combining ECG data with clinical information.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json

# Add parent directory to path to import cardioinsight_ai
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import CardioInsight AI
from cardioinsight_ai.main import CardioInsightAI

def main():
    """
    Main function.
    """
    print("CardioInsight AI - Multimodal Analysis Example")
    
    # Initialize system
    system = CardioInsightAI()
    
    # Generate synthetic ECG data for demonstration
    print("Generating synthetic ECG data...")
    ecg_data = generate_synthetic_ecg()
    
    # Set metadata
    metadata = {
        'sampling_rate': 250,  # Hz
        'patient_id': '12345',
        'recording_date': '2023-06-15'
    }
    
    # Prepare clinical data
    clinical_data = {
        "demographics": {
            "age": 65,
            "gender": "male",
            "height": 175,  # cm
            "weight": 80,   # kg
            "bmi": 26.1
        },
        "symptoms": [
            "chest_pain",
            "shortness_of_breath",
            "palpitations"
        ],
        "medical_history": [
            "hypertension",
            "diabetes_type_2",
            "previous_mi"
        ],
        "medications": [
            "aspirin",
            "metoprolol",
            "atorvastatin",
            "metformin"
        ],
        "lab_results": {
            "troponin_i": 0.8,  # ng/mL
            "ck_mb": 12.5,      # U/L
            "bnp": 450,         # pg/mL
            "potassium": 4.2,   # mEq/L
            "sodium": 138,      # mEq/L
            "creatinine": 1.1   # mg/dL
        },
        "vital_signs": {
            "heart_rate": 88,           # bpm
            "blood_pressure": "145/90",  # mmHg
            "respiratory_rate": 18,      # breaths/min
            "oxygen_saturation": 96      # %
        }
    }
    
    # Analyze ECG with multimodal fusion
    print("Performing multimodal analysis...")
    results = system.analyze_with_multimodal(ecg_data, clinical_data, metadata)
    
    # Print results
    print("\nMultimodal Analysis Results:")
    print(f"Prediction: {results['prediction']}")
    print(f"Confidence: {results['confidence']}")
    print(f"Model Type: {results['model_type']}")
    
    # Plot ECG data
    plt.figure(figsize=(12, 6))
    plt.plot(ecg_data[:1000])  # Plot first 1000 samples
    plt.title('Synthetic ECG Data')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig('synthetic_ecg_multimodal.png')
    print("ECG plot saved as 'synthetic_ecg_multimodal.png'")
    
    # Save clinical data
    with open('clinical_data.json', 'w') as f:
        json.dump(clinical_data, f, indent=2)
    print("Clinical data saved as 'clinical_data.json'")
    
    # Save results
    system.save_results(results, "multimodal_analysis_results.json")
    print("Results saved as 'multimodal_analysis_results.json'")
    
    # Shutdown system
    system.shutdown()
    
def generate_synthetic_ecg(duration=10, sampling_rate=250, num_leads=12):
    """
    Generate synthetic ECG data for demonstration.
    
    Parameters:
    -----------
    duration : float
        Duration in seconds. Default is 10.
    sampling_rate : int
        Sampling rate in Hz. Default is 250.
    num_leads : int
        Number of leads. Default is 12.
        
    Returns:
    --------
    ecg_data : array-like
        Synthetic ECG data.
    """
    # Number of samples
    n_samples = int(duration * sampling_rate)
    
    # Time array
    t = np.arange(n_samples) / sampling_rate
    
    # Generate synthetic ECG for each lead
    ecg_data = np.zeros((n_samples, num_leads))
    
    # Simulate abnormal ECG (e.g., myocardial infarction)
    # This is a simplified simulation for demonstration purposes
    
    for lead in range(num_leads):
        # Base frequency and phase
        base_freq = 1.2  # ~72 bpm
        phase = lead * 0.2
        
        # QRS complex
        qrs = np.sin(2 * np.pi * base_freq * t + phase) * np.exp(-((t % (1/base_freq) - 0.1) ** 2) / 0.001)
        
        # P wave
        p_wave = 0.2 * np.sin(2 * np.pi * base_freq * t + phase - 0.2) * np.exp(-((t % (1/base_freq) - 0.05) ** 2) / 0.002)
        
        # T wave (inverted for some leads to simulate MI)
        if lead in [0, 1, 2]:  # Leads I, II, III
            t_wave = -0.3 * np.sin(2 * np.pi * base_freq * t + phase + 0.3) * np.exp(-((t % (1/base_freq) - 0.3) ** 2) / 0.003)
        else:
            t_wave = 0.3 * np.sin(2 * np.pi * base_freq * t + phase + 0.3) * np.exp(-((t % (1/base_freq) - 0.3) ** 2) / 0.003)
        
        # ST segment elevation for some leads
        if lead in [0, 1, 2, 6, 7]:  # Leads I, II, III, V1, V2
            st_elevation = 0.2 * np.ones_like(t)
            st_elevation *= np.exp(-((t % (1/base_freq) - 0.2) ** 2) / 0.005)
        else:
            st_elevation = np.zeros_like(t)
        
        # Combine waves
        ecg = qrs + p_wave + t_wave + st_elevation
        
        # Add noise
        noise = 0.05 * np.random.randn(n_samples)
        
        # Add baseline wander
        baseline = 0.1 * np.sin(2 * np.pi * 0.05 * t + lead)
        
        # Combine all components
        ecg_data[:, lead] = ecg + noise + baseline
        
    return ecg_data

if __name__ == "__main__":
    main()

