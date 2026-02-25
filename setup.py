from setuptools import setup, find_packages

setup(
    name="nnlc-tools",
    version="0.1.0",
    description="NNLC (Neural Network Lateral Control) training tools for openpilot",
    url="https://github.com/amzoo/openpilot-nnlc-tools",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24,<2.0",
        "pandas>=2.0,<3.0",
        "matplotlib>=3.7,<4.0",
        "tqdm>=4.65",
        "zstandard>=0.21",
        "paramiko>=3.0",
    ],
    entry_points={
        "console_scripts": [
            "nnlc-sync=nnlc_tools.sync_rlogs:main",
            "nnlc-extract=nnlc_tools.extract_lateral_data:main",
            "nnlc-score=nnlc_tools.score_routes:main",
            "nnlc-visualize=nnlc_tools.visualize_coverage:main",
        ],
    },
)
