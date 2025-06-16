#!/usr/bin/env python
"""
NeuroRiskLogic Setup Configuration
Professional ML portfolio project for neurodevelopmental risk assessment
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neurorisklogic",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered neurodevelopmental risk assessment system using clinical and sociodemographic features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/NeuroRiskLogic",
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "pre-commit>=3.6.0",
        ],
        "ml": [
            "mlflow>=2.9.2",
            "wandb>=0.16.1",
        ],
        "cloud": [
            "boto3>=1.34.11",
            "google-cloud-storage>=2.13.0",
            "azure-storage-blob>=12.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neurorisk-train=scripts.train_model:main",
            "neurorisk-generate=scripts.generate_synthetic_data:main",
            "neurorisk-evaluate=scripts.evaluate_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
        "data": ["feature_definitions.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/NeuroRiskLogic/issues",
        "Source": "https://github.com/yourusername/NeuroRiskLogic",
        "Documentation": "https://github.com/yourusername/NeuroRiskLogic/wiki",
    },
)