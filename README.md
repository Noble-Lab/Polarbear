# Polarbear
Polarbear translates between different single-cell data modalities

![This is an image](https://noble.gs.washington.edu/~ranz0/Polarbear/polarbear_schematic_v2.png)


install polarbear conda environment (in linux)

```
conda env create -f environment_polarbear.yml
conda activate polarbear
```

download data from [this link](https://noble.gs.washington.edu/~ranz0/Polarbear/data/) to ./data/

train/evaluate the semi-supervised Polarbear model:

```
bash run_polarbear.sh babel semi
bash run_polarbear.sh random semi
```

train/evaluate the Polarbear-coassay model (that only uses co-assay data into training):

```
bash run_polarbear.sh babel coassay
bash run_polarbear.sh random coassay
```

generate performance comparison plots between Polarbear and Polarbear co-assay
```
mkdir -p result/
Rscript bin/evaluate_polarbear.R
```

Reference: Ran Zhang, Laetitia Meng-Papaxanthos, Jean-Philippe Vert, William Stafford Noble. [Semi-supervised single-cell cross-modality translation using Polarbear](https://doi.org/10.1101/2021.11.18.467517)
