from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ShopSmart",
    version="0.1",
    author="Shashank Lakhaiayr",
    packages=find_packages(),
    install_requires = requirements,
)