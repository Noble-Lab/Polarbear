# Polarbear
Polarbear translates between different single-cell data modalities

install polarbear conda environment

```
conda env create -f environment.yml
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

Reference: [https://doi.org/10.1101/2021.11.18.467517](https://doi.org/10.1101/2021.11.18.467517)
