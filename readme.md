# Running Instructions

## Environment Setup

1. First, ensure you have Conda installed on your system. If not, please install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. Create and activate the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate llamagen
```
If you want to install the environment in a specific directory, you can use the following command:
```bash
conda env create -f environment.yml -p /path/to/your/env
```
If you meet several errors, you need try to install the certain package in the environment.yml file.
## Running the Script

1. Make sure the run.sh script has execute permissions:
```bash
chmod +x run.sh
```

2. Execute the script:
```bash
./run.sh
```

## Environment Configuration

The `environment.yml` file contains all necessary dependencies. Here's an example of what it includes:

```yaml
name: llamagen
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.8
  - pip
  - pip:
    - torch
    - transformers
    # Add other required packages here
```

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are properly installed
2. Check that your Python version matches the one specified in environment.yml
3. Verify that run.sh has proper execute permissions
4. Check the logs for any error messages

## Additional Notes
- In run.sh, you need to modify the project root path, conda environment path, dataset path, model path, etc.
- The script requires sufficient disk space and memory
- GPU support is recommended for optimal performance
- For detailed logs, check the output directory after running the script
# Test_llamagan
