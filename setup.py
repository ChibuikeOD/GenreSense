from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="genresense",
    version="0.1.0",
    description="Spotify library auditing and recommendation pipeline with Flask, dlt, and scikit-learn.",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    install_requires=[
        "dlt[postgres]>=1.0.0",
        "flask>=3.0.0",
        "pandas>=2.2.0",
        "python-dotenv>=1.0.1",
        "requests>=2.32.0",
        "scikit-learn>=1.5.0",
        "spotipy>=2.25.0",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
