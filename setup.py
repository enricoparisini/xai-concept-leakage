import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="xai-concept-leakage",
    version="1.0.0",
    author="Enrico Parisini",
    author_email="eparisini@turing.ac.uk",
    description="Leakage scores",
    long_description=long_description,
    license='MIT',
    long_description_content_type="text/markdown",
    url="https://github.com/enricoparisini/xai-concept-leakage",
    packages=setuptools.find_packages(),
        classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.7',
    install_requires=[
        "importlib-metadata>=4.8.2",
        "importlib-resources>=5.4.0",
        "ipykernel>=6.5.0",
        "ipython-genutils>=0.2.0",
        "ipython>=7.29.0",
        "ipywidgets>=7.6.5",
        "joblib>=1.1.0",
        "matplotlib-inline>=0.1.3",
        "matplotlib>=3.5.0",
        "notebook>=6.4.5",
        "numpy==1.26.0",
        "pytorch-lightning>=1.6.0,<2.0.0",
        "scikit-learn-extra>=0.2.0",
        "scikit-learn>=1.0.1",
        "seaborn>=0.11.2",
        "torch>=1.11.0,<=2.3.1",
        "torchmetrics>=0.6.2",
        "torchvision>=0.12.0",
        "mergedeep>=1.3.4",
        "prettytable>=3.8.0",
        "tensorboard>=2.13.0",
        "tensorboard-data-server>=0.6.1",
        "tensorflow>=2.12.0",
        "tensorflow-datasets>=4.8.0",
        "tensorflow-metadata>=1.12.0",
    ],
)


