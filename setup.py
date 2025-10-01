from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="catattack",
    version="0.1.0",
    author="Collinear AI",
    author_email="research@collinear.ai",
    description="Query-Agnostic Adversarial Triggers for Reasoning Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/collinear-ai/CatAttack",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "cloud": [
            "boto3>=1.26.0",
            "google-cloud-aiplatform>=1.25.0",
        ],
        "viz": [
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "catattack=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "catattack": ["prompts/*.py", "examples/*.yaml"],
    },
)
