from setuptools import setup, find_packages

with open("requirements.txt") as fp:
    requirements = fp.read().splitlines()

setup(name="rl4lms",
      packages=find_packages(),
      python_requires='>=3.7',
      install_requires=requirements)
