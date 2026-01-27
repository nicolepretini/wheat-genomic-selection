# Wheat Genomic Selection — Synthetic Panel

End-to-end genomic selection (GS) workflow for wheat using a synthetic but realistic breeding panel.  
The project focuses on **honest model validation**, **genomic estimated breeding values (GEBVs)**, and **breeding-relevant decision outputs** such as parent ranking and cross evaluation.

---

## Project motivation

Genomic selection is widely used in modern plant breeding to accelerate genetic gain by predicting the genetic merit of lines from marker data.  
Unlike many data science examples, GS models must be evaluated carefully to avoid **genetic relatedness leakage** and must produce outputs that breeders can actually use.

This project demonstrates:
- a standard **RR-BLUP (Ridge regression)** genomic prediction baseline
- leakage-aware cross-validation using **GroupKFold**
- practical outputs for **parent selection** and **cross planning**
- a deployable preprocessing + model pipeline

All data in this repository are **synthetic**, generated to mimic real wheat breeding panels.

---

## Data description

The synthetic dataset represents a typical genomic selection panel:

- ~300 wheat cultivars / advanced breeding lines
- ~3,900 SNP markers (biallelic, additive dosage)
- One quantitative trait (`trait_y`) representing a complex agronomic phenotype (e.g. yield proxy)

### Files (not all tracked on GitHub)
- `data/raw/genotypes.csv` — SNP dosage matrix (large, ignored)
- `data/processed/genotypes_qc_std.csv` — QC’ed & standardized genotypes (large, ignored)
- `data/processed/phenotypes_clean.csv` — cleaned phenotypes (tracked)

Large genotype matrices are intentionally excluded from version control.

---

## Repository structure
wheat-genomic-selection/
├── notebooks/
│ ├── 01_eda_genomic_selection.ipynb
│ ├── 02_snp_qc_pca.ipynb
│ └── 03_ridge_gs_modeling.ipynb
├── src/
│ └── (data generation / utilities)
├── outputs/
│ ├── tables/
│ └── models/
├── data/
│ ├── raw/ (ignored)
│ └── processed/ (large genotype files ignored)
├── README.md
├── requirements.txt
└── .gitignore

---

## Methods

### Genomic prediction model
We use **Ridge regression**, equivalent to **RR-BLUP**, assuming:
- additive marker effects
- many small-effect loci
- shrinkage of SNP effects

Predictions are interpreted as **Genomic Estimated Breeding Values (GEBVs)** — relative additive genetic merit for the target trait.

---

### Validation strategy: avoiding genetic leakage

Random K-fold cross-validation can strongly overestimate predictive ability in genomic data because genetically related lines may appear in both training and test sets.

To obtain a more realistic estimate of performance when predicting **new genetic backgrounds**, we use:

1. PCA on genotype data to capture population structure  
2. K-means clustering on PCA scores to define **genetic origin groups**  
3. **GroupKFold** cross-validation across these groups  

This approximates predicting lines from unseen breeding programs or genetic pools.

---

### Breeding-relevant outputs

#### 1. Cultivar ranking (GEBVs)
After fitting the final model on all available data, each cultivar receives a GEBV.  
Cultivars are ranked to support **parent selection**.

These values are:
- trait-specific
- relative (not absolute phenotypic predictions)

#### 2. Cross evaluation (expected progeny mean)
Under additive genetics, the expected mean breeding value of a cross A × B is approximated by the **mid-parent value**:

\[
\hat{G}_{A×B} = \frac{GEBV_A + GEBV_B}{2}
\]

This allows ranking of candidate crosses by **expected mean performance**.

---

## Results overview

The notebook `03_ridge_gs_modeling.ipynb` produces:

- comparison of **KFold vs GroupKFold** predictive ability (Pearson r, RMSE)
- GEBV distribution and top-parent selection thresholds
- ranked list of candidate crosses by expected mean
- a saved, reusable GS pipeline:

outputs/models/ridge_gs_pipeline.joblib


As expected for a complex quantitative trait, predictive ability is modest but stable, and performance under GroupKFold is lower than under random CV, reflecting realistic generalization.

---

## Applying the model to new genotypes

The saved pipeline includes:
- SNP imputation
- feature scaling
- Ridge regression coefficients

To score a new genotype:
- it must be encoded on the same SNP panel
- SNP columns must be aligned to the training data
- missing values are handled automatically by the pipeline

Predictions are **trait-specific GEBVs** and should be interpreted for ranking, not as exact phenotypic values.

---

## Planned extensions

### Cross variance via progeny simulation
Current cross evaluation compares only expected means.  
A planned extension will simulate progeny genotypes (e.g. F2 or DH approximation) to estimate:

- segregation variance
- probability of extreme progeny
- expected response under selection

This enables ranking crosses by both **mean** and **upper-tail performance**.

### Multi-trait genomic selection
Future steps will include:
- fitting separate models for multiple traits
- constructing a **selection index** combining GEBVs across traits
- optional multi-output regression models

---

## Reproducibility

### Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

###Jupyter
pip install ipykernel
python -m ipykernel install --user --name wheat-gs --display-name "Python (wheat-gs)"
jupyter notebook


Select kernel: Python (wheat-gs)

Notes on synthetic data

All data are synthetic and generated for learning and demonstration purposes.
The modeling, validation logic, and interpretation mirror real genomic selection workflows and can be directly transferred to real breeding datasets with consistent SNP coding.

Author

Nicole Pretini
Genetics · Agronomy · Data Science


---

## Multi-trait Genomic Selection & Cross Ranking

This project implements a full genomic selection workflow extending beyond single-trait prediction:

- Multi-trait GEBV estimation (yield, disease resistance, plant height)
- Construction of a weighted selection index reflecting breeding priorities
- Simulation of progeny genotypes (F2 / DH) from parental crosses
- Estimation of within-cross additive variance
- Cross ranking based on expected performance and upper-tail potential (P95)

The final output is a production-style decision table identifying optimal parental crosses under realistic trade-offs between traits and genetic variance. This mirrors how modern breeding programs prioritize crosses for both short-term gain and long-term genetic improvement.
