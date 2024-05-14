import setuptools

setuptools.setup(
    name="fguard",
    version="0.1.0",
    py_modules=[
        "fguard",
        "communication",
        "vision",
        "handler",
        "utils",
    ],
    install_requires=[
        "click==8.1.7",
        "eo-learn==1.5.2",
        "joblib==1.3.2",
        "numpy==1.26.3",
        "opencv-python==4.8.1.78",
        "python-dotenv==1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "fguard = fguard:main",
        ],
    },
)
