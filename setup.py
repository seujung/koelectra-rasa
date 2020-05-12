from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="koElectra-pytorch",
    version="0.1",
    description="Koelecta based intent / entity model",
    author="seujung",
    install_requires=[],
    packages=find_packages(exclude=["docs", "tests", "tmp", "data"]),
    python_requires=">=3",
    zip_safe=False,
    include_package_data=True,
)
