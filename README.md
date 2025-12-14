# SNACS-2025 — Graph Sampling for Node Classification

This repo evaluates **classic random graph sampling methods** from a **downstream node-classification** perspective.

We benchmark **10 samplers** on **8 graphs**, sampling **only the training nodes** under different budgets while keeping the **validation/test splits fixed on the original graph**. For each sampled graph we train two models:

- **MLP** (features only)
- **GCN** (features + graph structure)

We report **weighted F1** over **10 random seeds** and aggregate results using **average rank** to compare robustness across datasets and sample sizes.

**Key findings:**
- **Hybrid node–edge sampling** is the most consistent overall, followed by **Random Node–Edge** and **Random Node**.
- Citation graphs tend to favor **node-based** sampling; denser graphs benefit from **node–edge** strategies that balance coverage and connectivity.
- Performance generally improves with larger sample budgets; very small samples are less stable, especially for exploration-based sampling methods.

---

## Repository structure

- `run_experiment.py` — run the full experiment pipeline  
- `config.py` — all experiment hyperparameters (datasets, samplers,  training settings, seeds, etc.)  
- `models/` — model implementations  
- `samplers/` — sampler implementations  
- `helpers/` — utilities (data loading, metrics, logging, etc.)  
- `results/` — output `.csv` files with weighted F1 scores per dataset × model × sampling ratio (averaged over 10 seeds)

---

## Setup

### 1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
``` 

### 2. Upgrade pip
```bash
pip install --upgrade pip
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the experiments
```bash
python run_experiment.py
```

The specified in the config file will be downloaded atuomatically under a new ./data folder. Results will be saved automatically to the results/ directory as .csv files.
