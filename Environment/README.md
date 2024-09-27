To set the environment properly and avoid any conflicts follow the steps:
1- Run the following command line in the terminal, it will create a Conda environment with the name physgnn:
conda env create -f GNNs-BreastCompression/Environment/conda_requirements.yml
2- Activate physgnn env., then install the pip requirements by running the following command:
conda activate physgnn
pip install -r GNNs-BreastCompression/Environment/pip_requirements.txt
3- Run torch-geometric.ipynb notebook using the physgnn env. as a kernel. To install torch-geometric and other important packages from a wheel. The rest of the notebook is to verify that the packages were installed correctly.

# Note: Note: The previous versions of the packages are compatible with CUDA 11.7. They were also tested with CUDA 11.8 and 12.4, and no issues were encountered.
