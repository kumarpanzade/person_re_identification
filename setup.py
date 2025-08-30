from setuptools import setup, find_packages

setup(
    name="squash_player_tracker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",  # For YOLOv8
        "torchreid==0.2.5",    # For OSNet person re-identification
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "deep-sort-realtime>=1.3.2",
    ],
    python_requires=">=3.8",
)