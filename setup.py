"""
KrystalSDK - Adaptive Intelligence for Any Project
"""

from setuptools import setup, find_packages

setup(
    name="krystal-sdk",
    version="0.1.0",
    description="Drop-in adaptive intelligence with TD-learning, phase transitions, and swarm optimization",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="KrystalStack Team",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "full": ["numpy", "fastapi", "uvicorn"],
        "dev": ["pytest", "black", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "krystal=python.cli:main",
            "krystal-dashboard=python.web_dashboard:run_server",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
