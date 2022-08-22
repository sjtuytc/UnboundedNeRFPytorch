import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setuptools.setup(
    name="dvlab_nerf", # Replace with your own username
    version="0.0.1",
    author="Zelin Zhao",
    author_email="sjtuytc@gmail.com",
    description="Code for large-scale neural radiance fields.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dvlab-research/BlockNeRFPytorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)