from setuptools import setup, find_packages

setup(
    name="ott",
    version="0.1.0",
    description="Optimal Transport Tools - Local Version",
    packages=find_packages(),
    install_requires=[
        "jax>=0.7.0",
        "jaxlib>=0.7.0",
        "numpy>=1.20.0,<2.0.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
) 