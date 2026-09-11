"""
Setup script for Prompt Injection Defense System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prompt-injection-defense",
    version="3.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="5-Layer Pipeline for Prompt Injection Detection and Mitigation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/prompt-injection-defense",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.2.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "xgboost>=1.7.0",
        "joblib>=1.2.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
        "shap>=0.41.0",
        "lime>=0.2.0.1",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "tqdm>=4.65.0",
        "aiofiles>=23.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.1.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ],
        "llm": [
            "ollama>=0.1.0",
            "openai>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prompt-defense=main:main",
            "prompt-defense-api=run_api:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
