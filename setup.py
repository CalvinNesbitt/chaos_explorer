from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="chaos_explorer",
    version="0.1",
    description="Computations in nonlinear dynamics.",
    author="Calvin Nesbitt",
    author_email="cf.nesbitt95@gmail.com",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    install_requires=requirements,
)
