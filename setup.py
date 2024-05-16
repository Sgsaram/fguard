import setuptools

setuptools.setup(
    name="fguard",
    version="1.1.1",
    url="https://forestguardian.ru/",
    author="Ivan Gronsky",
    description="Detect deforestation on area over a given period of time",
    long_description=open("README.MD").read(),
    long_description_content_type='text/markdown',
    license="GPL-3.0",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    packages=setuptools.find_packages(),
    install_requires=[
        "click==8.1.7",
        "eo-learn==1.5.2",
        "joblib==1.3.2",
        "numpy==1.26.3",
        "opencv-python==4.8.1.78",
        "platformdirs==4.2.1",
        "python-dotenv==1.0.0",
        "requests==2.31.0",
        "tqdm==4.66.2",
        "torch==2.3.0",
        "torchvision==0.18.0",
    ],
    entry_points={
        "console_scripts": [
            "fguard = fguard.cli:main",
        ],
    },
)
