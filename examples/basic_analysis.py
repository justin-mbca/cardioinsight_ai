"""
CardioInsight AI - Basic Analysis Example

This example demonstrates how to use the CardioInsight AI system for basic ECG analysis.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import cardioinsight_ai
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import CardioInsight AI
from cardioinsight_ai.main import CardioInsightAI

def main():
    """
    Main function.
    """
    print("CardioInsight AI - Basic Analysis Example")
    
    # Initialize system
    system = CardioInsightAI()
    
    # Generate synthetic ECG data for demonstration
    print("Generating synthetic ECG data...")
    ecg_data = generate_synthetic_ecg()
    
    # Set metadata
    metadata = {
        'sampling_rate': 250,  # Hz
        'patient_id': '12345',
        'age': 65,
        'gender': 'male'
    }
    
    # Analyze ECG
    print("Analyzing ECG data...")
    results = system.analyze_ecg(ecg_data, metadata, use_dl=True, explain=True)
    
    # Print results
    print("\nAnalysis Results:")
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
    plt.savefig('synthetic_ecg.png')
    print("ECG plot saved as 'synthetic_ecg.png'")
    
    # Save results
    system.save_results(results, "analysis_results.json")
    print("Results saved as 'analysis_results.json'")
    
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
    
    for lead in range(num_leads):
        # Base frequency and phase
        base_freq = 1.2  # ~72 bpm
        phase = lead * 0.2
        
        # QRS complex
        qrs = np.sin(2 * np.pi * base_freq * t + phase) * np.exp(-((t % (1/base_freq) - 0.1) ** 2) / 0.001)
        
        # P wave
        p_wave = 0.2 * np.sin(2 * np.pi * base_freq * t + phase - 0.2) * np.exp(-((t % (1/base_freq) - 0.05) ** 2) / 0.002)
        
        # T wave
        t_wave = 0.3 * np.sin(2 * np.pi * base_freq * t + phase + 0.3) * np.exp(-((t % (1/base_freq) - 0.3) ** 2) / 0.003)
        
        # Combine waves
        ecg = qrs + p_wave + t_wave
        
        # Add noise
        noise = 0.05 * np.random.randn(n_samples)
        
        # Add baseline wander
        baseline = 0.1 * np.sin(2 * np.pi * 0.05 * t + lead)
        
        # Combine all components
        ecg_data[:, lead] = ecg + noise + baseline
        
    return ecg_data

if __name__ == "__main__":
    main()

