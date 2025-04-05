#!/usr/bin/env python
"""Setup script for SentinelDocs."""

from setuptools import setup, find_packages
import os


def read_requirements():
    """Read the requirements.txt file and return a list of requirements."""
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="sentineldocs",
    version="1.0.0",
    description="AI-powered document analysis and semantic search tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mayur",
    url="https://github.com/MayurHulke/sentineldocs",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    entry_points={
        "console_scripts": [
            "sentineldocs=sentineldocs.app:main",
        ],
    },
) 