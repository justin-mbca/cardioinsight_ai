"""
CardioInsight AI - Setup Script

This script is used to install the CardioInsight AI system and its dependencies.
"""

from setuptools import setup, find_packages

setup(
    name="cardioinsight_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.23.0",
        "tensorflow>=2.4.0",
        "tensorflow-model-optimization>=0.5.0",
        "keras>=2.4.0",
        "wfdb>=3.3.0",
        "neurokit2>=0.1.0",
        "biosppy>=0.7.0",
        "requests>=2.25.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "PyWavelets>=1.1.0",
        "tqdm>=4.50.0",
        "joblib>=1.0.0",
        "h5py>=3.1.0",
        "seaborn>=0.11.0",
    ],
    author="CardioInsight AI Team",
    author_email="info@cardioinsight.ai",
    description="Multi-modal Intelligent ECG Analysis System",
    keywords="ECG, AI, healthcare, cardiology",
    url="https://github.com/cardioinsight/cardioinsight-ai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "cardioinsight=cardioinsight_ai.main:main",
        ],
    },
)

