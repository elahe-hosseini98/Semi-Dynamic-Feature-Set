from setuptools import setup, find_packages

setup(
    name="sdfs",
    version="1.0.1",  
    author="Elaheh Hosseini",
    author_email="elahe.s.hs98@gmail.com",
    description="A novel approach for Semi-Dynamic Feature Sets (SDFS)",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/elahe-hosseini98/Semi-Dynamic-Feature-Set",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "pandas"  
    ],
    package_data={
        'examples': ['winequalityN.csv'],  
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
