from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_ld_blocks(
    n_snps: int,
    block_size: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create an AR(1)-like covariance per LD block: cov(i,j)=rho^|i-j|.
    Returns block membership index for each SNP.
    """
    n_blocks = int(np.ceil(n_snps / block_size))
    block_ids = np.repeat(np.arange(n_blocks), block_size)[:n_snps]
    return block_ids


def simulate_genotypes(
    n_individuals: int = 300,
    n_snps: int = 39_000,
    n_pops: int = 3,
    pop_shift_sd: float = 0.6,
    block_size: int = 200,
    ld_rho: float = 0.85,
    missing_rate: float = 0.02,
    rng_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Simulate 0/1/2 dosages with:
    - population structure (n_pops)
    - SNP allele frequency variation across pops
    - LD-like blocks via shared latent factors within blocks
    - missingness
    """
    rng = np.random.default_rng(rng_seed)

    # Individuals and population assignment
    indiv_ids = [f"CULT_{i+1:03d}" for i in range(n_individuals)]
    pops = pd.Series(rng.integers(0, n_pops, size=n_individuals), index=indiv_ids, name="pop")

    # Base allele frequencies: skewed to low-ish MAF (Beta)
    # Choose base p in (0.02, 0.98), then enforce min MAF later if you want.
    base_p = rng.beta(a=0.8, b=2.5, size=n_snps)
    base_p = np.clip(base_p, 0.02, 0.98)

    # Population-specific shifts on logit scale
    # p_pop = logistic(logit(p) + shift_pop)
    logit_p = np.log(base_p / (1.0 - base_p))
    pop_shifts = rng.normal(0.0, pop_shift_sd, size=(n_pops, n_snps))
    p_by_pop = logistic(logit_p[None, :] + pop_shifts)
    p_by_pop = np.clip(p_by_pop, 0.01, 0.99)

    # LD blocks via latent factors: for each block, generate a latent z per individual
    block_ids = make_ld_blocks(n_snps=n_snps, block_size=block_size, rho=ld_rho, rng=rng)
    n_blocks = block_ids.max() + 1

    # For each block and individual: latent effect that induces correlation among SNPs in same block
    z = rng.normal(0.0, 1.0, size=(n_individuals, n_blocks))

    # Within each SNP, build individual-specific p_ij around pop p with latent factor
    # p_ij = logistic(logit(p_pop) + w_snp * z_indiv_block)
    w = rng.normal(loc=0.0, scale=0.8, size=n_snps)  # SNP loading on block latent
    snp_names = [f"SNP_{j+1:05d}" for j in range(n_snps)]

    G = np.empty((n_individuals, n_snps), dtype=np.float32)

    for j in range(n_snps):
        b = block_ids[j]
        # Choose each individual's pop-specific allele frequency for this SNP
        p0 = p_by_pop[pops.values, j]
        lp0 = np.log(p0 / (1.0 - p0))
        lp_ij = lp0 + w[j] * z[:, b] * 0.35  # 0.35 controls within-block correlation strength
        p_ij = logistic(lp_ij)
        p_ij = np.clip(p_ij, 0.001, 0.999)

        # Draw genotype as Binomial(2, p_ij)
        G[:, j] = rng.binomial(2, p_ij).astype(np.float32)

    # Inject missingness
    if missing_rate > 0:
        mask = rng.random(G.shape) < missing_rate
        G = G.astype("float32")
        G[mask] = np.nan

    geno_df = pd.DataFrame(G, index=indiv_ids, columns=snp_names)

    # SNP metadata
    snp_meta = pd.DataFrame({
        "snp": snp_names,
        "block_id": block_ids,
        "base_p": base_p,
        "loading_w": w,
    })

    return geno_df, snp_meta, pops


def simulate_phenotype(
    geno_df: pd.DataFrame,
    pops: pd.Series,
    n_qtl: int = 200,
    h2: float = 0.40,
    pop_effect_sd: float = 0.3,
    rng_seed: int = 43,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate quantitative phenotype:
    y = genetic + pop_effect + noise
    genetic = sum_{qtl} beta_j * standardized_dosage_j
    Noise variance set to achieve approximate narrow-sense h2.
    """
    rng = np.random.default_rng(rng_seed)

    n_indiv, n_snps = geno_df.shape
    snp_names = geno_df.columns.to_numpy()

    # Choose QTLs
    qtl_idx = rng.choice(n_snps, size=min(n_qtl, n_snps), replace=False)
    qtl_snps = snp_names[qtl_idx]

    # Impute temporarily for phenotype simulation only (mean impute)
    X = geno_df[qtl_snps].to_numpy(dtype=np.float32)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    # Standardize QTL genotypes
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Effect sizes (many small effects)
    beta = rng.normal(0.0, 1.0, size=X.shape[1])
    beta = beta / np.linalg.norm(beta)  # normalize overall genetic scale

    genetic = X @ beta

    # Population effect (captures structure)
    pop_ids = pops.values
    pop_levels = np.unique(pop_ids)
    pop_effects = rng.normal(0.0, pop_effect_sd, size=len(pop_levels))
    pop_term = pop_effects[pop_ids]

    # Set noise to achieve approximate h2 wrt genetic component only
    var_g = np.var(genetic)
    if var_g <= 1e-12:
        var_g = 1.0

    # If you include pop_term as "non-genetic", keep h2 relative to (genetic + noise)
    # We'll target: h2 = var_g / (var_g + var_e) => var_e = var_g*(1-h2)/h2
    var_e = var_g * (1.0 - h2) / max(h2, 1e-6)
    noise = rng.normal(0.0, np.sqrt(var_e), size=n_indiv)

    y = genetic + pop_term + noise

    pheno_df = pd.DataFrame({
        "cultivar": geno_df.index,
        "trait_y": y,
        "pop": pops.values,
    })

    qtl_meta = pd.DataFrame({
        "qtl_snp": qtl_snps,
        "beta": beta,
    })

    return pheno_df, qtl_meta


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # --- Config: edit here if you want ---
    cfg = {
        "n_individuals": 300,
        "n_snps": 39_000,
        "n_pops": 3,
        "pop_shift_sd": 0.6,
        "block_size": 200,
        "ld_rho": 0.85,
        "missing_rate": 0.02,
        "n_qtl": 200,
        "h2": 0.40,
        "pop_effect_sd": 0.30,
        "seed_geno": 42,
        "seed_pheno": 43,
    }

    geno_df, snp_meta, pops = simulate_genotypes(
        n_individuals=cfg["n_individuals"],
        n_snps=cfg["n_snps"],
        n_pops=cfg["n_pops"],
        pop_shift_sd=cfg["pop_shift_sd"],
        block_size=cfg["block_size"],
        ld_rho=cfg["ld_rho"],
        missing_rate=cfg["missing_rate"],
        rng_seed=cfg["seed_geno"],
    )

    pheno_df, qtl_meta = simulate_phenotype(
        geno_df=geno_df,
        pops=pops,
        n_qtl=cfg["n_qtl"],
        h2=cfg["h2"],
        pop_effect_sd=cfg["pop_effect_sd"],
        rng_seed=cfg["seed_pheno"],
    )

    # Save files
    geno_path = os.path.join(raw_dir, "genotypes.csv")
    pheno_path = os.path.join(raw_dir, "phenotypes.csv")
    snp_meta_path = os.path.join(raw_dir, "snp_metadata.csv")
    qtl_meta_path = os.path.join(raw_dir, "qtl_metadata.csv")
    cfg_path = os.path.join(raw_dir, "synthetic_config.json")

    # Genotypes can be big; CSV is simplest. Parquet would be smaller but requires pyarrow.
    geno_df.to_csv(geno_path, index=True)
    pheno_df.to_csv(pheno_path, index=False)
    snp_meta.to_csv(snp_meta_path, index=False)
    qtl_meta.to_csv(qtl_meta_path, index=False)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("SAVED:")
    print(" -", geno_path)
    print(" -", pheno_path)
    print(" -", snp_meta_path)
    print(" -", qtl_meta_path)
    print(" -", cfg_path)
    print("\nSHAPES:")
    print(" genotypes:", geno_df.shape, "(rows=cultivars, cols=SNPs)")
    print(" phenotypes:", pheno_df.shape)


if __name__ == "__main__":
    main()
