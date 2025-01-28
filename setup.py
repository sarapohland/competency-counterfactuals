from setuptools import setup, find_packages

requirements = ['accelerate',
                'anomalib',
                'huggingface-hub',
                'numpy',
                'lpips',
                'matplotlib',
                'natsort',
                'opencv-python',
                'pandas',
                'Pillow',
                'pytorch-ood',
                'pyyaml',
                'scikit-image',
                'scikit-learn',
                'scipy',
                'seaborn',
                'tabulate',
                'torch',
                'torchvision',
                'torch-fidelity',
                'tqdm',
                'transformers'] 

setup(
    name="src",
    version="1.0.0",
    description="Probabilistic and Reconstruction-based Competency Estimation (PaRCE)",
    packages=find_packages(),
    install_requires=requirements
)