## Folder Structure

The following is the required folder structure for this project. If any of the folders are missing, please create them:
```
GNNs-BreastCompression/
└── Hold-out/
    ├── dataset/
    │   ├── a/
    │   ├── b/
    │   ├── c/
    │   ├── d/
    │   ├── e/
    │   ├── f/
    │   ├── g/
    │   ├── h/
    │   ├── i/
    │   └── j/
    ├── dataset_pickle/
    └── Results_Hold-out/
        └── csv/
            ├── test/
            └── val/
```
## Dataset Generation and running the experiment:
1. Run `dataset_full.py`, generating graph data for the 10 distinct batches of directions.
2. Run `pg_dataset.py`, transforming the generated graph data into NetworkX graphs, converting them to PyTorch Geometric format, and storing the result as a .pickle file. For each batch [a,b,c,...,j] this is done separately by uncommenting the corresponding path.
4. Run `main.py`, to start the training. The 3rd configuration of the PhysGNN models is used.
5. Run `reproduce.py`, to reproduce the quantitative results on the test set.
