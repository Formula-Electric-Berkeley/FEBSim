from setuptools import setup, find_packages

setup(
    packages=find_packages(),  # Automatically find packages in the current directory
    install_requires=[
        'numpy>=1.19.3,<2.0',  # Adjust based on your needs
        'pandas>=1.1,<2.0',
        'matplotlib>=3.3,<4.0',
        'scipy>=1.5,<2.0',
        # Add any other dependencies you need
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # License type (change if needed)
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the Python version requirement
)
