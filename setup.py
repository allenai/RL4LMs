from setuptools import setup, find_packages

with open("requirements.txt") as fp:
    requirements = fp.read().splitlines()

setup(
    name="rl4lms",
    version="0.2.1",
    description="A library for training language models (LM) using RL",
    author="Rajkumar Ramamurthy, Prithviraj Ammanabrolu",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
    url="https://github.com/allenai/RL4LMs",
)
