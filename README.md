# GNNs-BreastCompression

- The presented code in this repository is the implementation of PhysGNN for modelling breast mammographic compression, available at:
- Please cite as:

- The breast geometry used in this study, 'UncompressedBreast3,' was obtained from a publicly available dataset on Zenodo, available at: https://zenodo.org/records/4529852
- The mesh generation tool used is Pygalmesh implemented by Schlömer et al, available at: https://github.com/meshpro/pygalmesh
- The FEA-baseline using NiftySim and Phantom Reconstruction was adopted from the work of García et al, available at: https://github.com/eloygarcia/niftysim & https://github.com/eloygarcia/RadboudCompression
- The PhysGNN implementation used in this work was originally developed by Salehi et al, available at: https://github.com/YasminSalehi/PhysGNN


**To set up the environment, please refer to the instructions in the [Environment](./Environment) folder.**

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

## Preprocessing the the uncompressed phantom:

**Run the following command with the right paths, in the public dataset there is a metadata csv file that will be needed:**

***The isotropic spacing that was used is 0.273***

`python GNNs-BreastCompression/preprocessing.py <dicom_folder> <csv_file> <output_folder> <isotropic_spacing>`

## Running the simulations:

   
