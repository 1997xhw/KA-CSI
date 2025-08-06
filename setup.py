from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ka-csi",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="CSI WiFi Activity Recognition using KAN and Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/KA-CSI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
        "tensorboard": ["tensorboard>=2.8.0"],
    },
    entry_points={
        "console_scripts": [
            "ka-csi-train=train:main",
            "ka-csi-visualize=visualize_results:main",
        ],
    },
) 