from pathlib import Path
from setuptools import setup, find_packages

setup_path = Path(__file__).parent

setup(
    name = "gaustudio",
    packages=find_packages()
)