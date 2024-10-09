# GNNs-BreastCompression

- The breast geometry used in this study, 'UncompressedBreast3,' was obtained from a publicly available dataset on Zenodo, available at: https://zenodo.org/records/4529852
- The mesh generation tool used is Pygalmesh implemented by Schlömer et al, available at: https://github.com/meshpro/pygalmesh
- The FEA-baseline using NiftySim and Phantom Reconstruction was adopted from the work of García et al, available at: https://github.com/eloygarcia/niftysim & https://github.com/eloygarcia/RadboudCompression
- The PhysGNN implementation used in this work was originally developed by Salehi et al, available at: https://github.com/YasminSalehi/PhysGNN

## Folder Structure:
**The following is the required folder structure for this project. If any of the folders are missing, please create them:**
```
GNNs-BreastCompression/
├── Data_Generator/
├── Environment/
├── FEA-simulations/
├── Hold-out/
├── LODO/
├── process-compressed/
├── uncompressed_nrrd/
├── mesh_data.ipynb
├── niftysim.ipynb
├── preprocessing.ipynb
└── qualitative.ipynb
```

## Environment
**To set up the environment, please refer to the instructions in the [Environment](./Environment) folder.**

## Dockers
**To build NiftySim Docker, please follow the instructions:**

```
git clone https://github.com/eloygarcia/niftysim.git
cd niftysim
docker build -t niftysim:2.5 .
```

**To build Reconstruct Phantom Docker, please follow the instructions:**

```
git clone https://github.com/eloygarcia/RadboudCompression.git
cd "RadboudCompression/Phantom Reconstruction/Reconstruct Image"
docker build -t reconstruct-image .
```
# Preprocessing
## Preprocessing the the uncompressed phantom:

**Run the following command with the right paths, in the public dataset there is a metadata CSV file that will be needed:**

***The isotropic spacing that was used is 0.273***

`python GNNs-BreastCompression/preprocessing.py <dicom_folder> <csv_file> <output_folder> <isotropic_spacing>`

# Running the FEA-simulations
**Make sure to update the path as prompted in the notebook `niftysim.ipynb`. This notebook is for mesh generation, runs NiftySim simulations, and reconstructs the phantom image. Incremental simulations will be achieved by adjusting the thickness and offset of the plates with each run and manually saving the output displacements from the Niftysim Docker output.**

# Data Extraction
**Run `mesh_data.ipynb` to extract the data from the uncompressed mesh, and generate the random force directions** 

# Experiments
**For Hold-out experiment, please refer to the instructions in the [Hold-out](./Hold-out) folder.**
**For Leave-one-deforamtion-out experiment, please refer to the instructions in the [LODO](./LODO) folder.**

# Processing PhysGNN output
**To produce qualitative results of `LODO` experiment, run `qualitative.ipynb` and make sure to update the path as prompted.**

