"""Setup script for pygent."""
from setuptools import find_packages, setup

setup(
    name="pygent-ai",
    version="0.1.0",
    description="A Python framework for building LLM-powered agents with modular state, tools, and MCP support.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="pygent-ai",
    url="https://github.com/your-org/pygent",
    license="Apache-2.0",
    packages=find_packages(exclude=["examples*", "tests*", "MCPs*", "docs*"]),
    python_requires=">=3.11",
    install_requires=[
        "fastmcp>=2.14.5",
        "openai>=2.17.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

