import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GRML_analyses",
    version="1.0.0",
    author="M. Isabel Agea",
    author_email="agealorm@vscht.cz",
    description="A Python library to predict glucocorticoid active ligands",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Iagea/GRML_analyses",
    packages=setuptools.find_packages(exclude=["tests.*", "tests"]),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Database",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.11',
)