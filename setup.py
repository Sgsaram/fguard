import setuptools

setuptools.setup(
    name="fguard",
    version="1.0.0",
    url="https://forestguardian.ru/",
    author="Ivan Gronsky",
    description="Detect deforestation on area over a given period of time",
    long_description=open("README.MD").read(),
    long_description_content_type='text/markdown',
    license="GPL-3.0",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.3',
    packages=["fguard"],
    install_requires=[
        "click==8.1.7",
        "eo-learn==1.5.2",
        "joblib==1.3.2",
        "numpy==1.26.3",
        "opencv-python==4.8.1.78",
        "platformdirs==4.2.1",
        "python-dotenv==1.0.0",
        "tqdm==4.66.2",
    ],
    entry_points={
        "console_scripts": [
            "fguard = fguard.cli:main",
        ],
    },
    include_package_data=True,
    package_data={"": ["assets/*.pth"]},
)
