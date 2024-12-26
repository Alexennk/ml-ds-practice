from setuptools import setup, find_packages
from pathlib import Path


def read_requirements(filename):
    return Path(filename).read_text().splitlines()


test_deps = [
    "pip>=24.3.0",
    "flake8>=6.0.0",
    "black>=22.6.0",
]

extras = {"test": test_deps}

setup(
    name="innowise-ml-internship-alexennk",
    version="1.0.5",
    author="Aleksey Senkin",
    author_email="aleksey.senkin@innowise.com",
    description="A package with scripts for ml-ds-project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    extras_require=extras,
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
