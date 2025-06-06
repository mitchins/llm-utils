from setuptools import setup, find_packages

setup(
    name='llm-utils',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'torch',
        # add any other deps
    ],
)