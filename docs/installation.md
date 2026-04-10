# Installation

## From PyPI

Once the package is published, install it with:

```bash
pip install reserbugs
```

## From The Repository

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/JJ-Lab/reserBUGS.git
cd reserBUGS
pip install -e .
```

To include documentation tools:

```bash
pip install -e ".[docs]"
```

To include optional scoring dependencies:

```bash
pip install -e ".[scoring]"
```

## Conda Environment

The repository also includes an `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate reserBUGS
pip install -e .
```

## Build The Documentation Locally

Install the documentation dependencies and run:

```bash
mkdocs serve
```

Then open the local URL printed by MkDocs.

To build the static site:

```bash
mkdocs build
```

## Python Support

reserBUGS requires Python 3.10 or newer.
