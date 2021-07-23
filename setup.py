import setuptools


with open("README.md", "r") as f:
  readme = f.read()
with open("featgraph/__init__.py", "r") as f:
  version = f.read().split("__version__ = \"", 1)[-1].split("\"", 1)[0]


setuptools.setup(
  name="featgraph",
  version=version,
  author="Marco Tiraboschi, Giulia Clerici",
  author_email="marco.tiraboschi@unimi.it, giulia.clerici@unimi.it",
  description="featgraph",
  long_description=readme,
  long_description_content_type="text/markdown",
  url="https://github.com/ChromaticIsobar/featgraph",
  packages=setuptools.find_packages(
    include=["featgraph", "featgraph.*"]
  ),
  include_package_data=True,
  setup_requires=[
    "wheel",
  ],
  install_requires=[
    "JPype1",
    "requests",
    "chromatictools",
    "sortedcontainers",
  ],
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires=">=3.5.0,<3.9.0",
)
