from pathlib import Path
from setuptools import setup, find_packages

setup_path = Path(__file__).parent

# Load dependencies from requirements.txt
requirements_path = setup_path / "requirements.txt"
if requirements_path.exists():
    dependencies = requirements_path.read_text().split("\n")
else:
    raise ValueError("requirements.txt doesn't exist. Please create the file and specify project dependencies.")

setup(
    name = "gaustudio",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=dependencies,
    long_description=(setup_path / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/hugoycj/gaustudio/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)