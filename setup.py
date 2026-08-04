# Copyright (C) 2026 Applied Intuition, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

setup(
    name="nord",
    version="1.0.0",
    description="NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning (CVPR 2026)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ishaan Rawal, Shubh Gupta, Yihan Hu, Wei Zhan",
    url="https://github.com/Applied-Intuition-Open-Source/nord",
    packages=find_packages(include=["nord", "nord.*"]),
    package_data={"nord": ["data/vocab.pkl"]},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.1",
        "numpy>=1.23",
        "Pillow>=9.0",
        "requests>=2.28.0",
        "huggingface_hub>=0.24.0",
    ],
    extras_require={
        "serve": ["vllm>=0.4.0", "transformers>=4.45.0"],
        "viz": ["matplotlib>=3.7.0"],
        "eval": ["tqdm>=4.64.0", "pandas>=1.5.0"],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)