AI-Augmented Computational Design of Deep Eutectic Solvents (DES)


Automated computational pipeline for property-driven design of Deep Eutectic Solvents (DES) integrating quantum chemical calculations (GFN2-xTB) with machine learning. This pipeline transforms raw chemical structures into accurate property predictions through a reproducible four-phase workflow.

Figure : Computational Pipeline Architecture
┌─────────────────────────────────────────────────────────────────────┐
│                      COMPUTATIONAL PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1: Data Preparation                                          │
│  └─ Script 1: raw_dataset.csv → cleaned_dataset_CORRECTED.csv       │
│     (RDKit validation, molecular properties)                        │
│                                                                     │
│  Phase 2: Quantum Geometry Optimization                             │
│  └─ Script 2: SMILES → 3D structures → GFN2-xTB optimization        │
│     (OpenBabel MMFF94 → xTB geometry optimization)                  │
│                                                                     │
│  Phase 3: Quantum Descriptor Calculation                            │
│  └─ Script 3: optimized structures → quantum descriptors            │
│     (xTB single-point: HOMO, LUMO, dipole, solvation, etc.)         │
│                                                                     │
│  Phase 4: Feature Engineering                                       │
│  └─ Script 4: quantum descriptors + composition → ML features       │
│     (Weighted averages, differences, stoichiometric features)       │
│                                                                     │
│  Output: des_features_final.csv → Machine Learning Models           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Prerequisites
Python 3.9 or higher


# Clone repository
git clone https://github.com/yourusername/DES-QSPR-Pipeline.git
cd DES-QSPR-Pipeline

# Install Python dependencies
pip install -r requirements.txt

# Install external dependencies separately:
# xTB: https://github.com/grimme-lab/xtb/releases
# OpenBabel: conda install -c conda-forge openbabel
# RDKit: conda install -c conda-forge rdkit



# Execute sequentially (using example data)
python scripts/01_data_preparation.py --input data/example_dataset.csv
python scripts/02_geometry_optimization.py
python scripts/03_calculatedescriptors.py
python scripts/04_feature_engineering.py

# Output will be in: des_features_final.csv


📊 Example Output
After running the complete pipeline, you'll get:

Phase 1 Outputs:
---------------
cleaned_dataset_CORRECTED.csv - Validated DES compositions
unique_molecules_CORRECTED.txt - All unique components
molecular_descriptors.csv - 2D molecular properties

Phase 2 Outputs:
---------------
optimized_structures/ - XYZ files for each component
optimization_results.csv - Optimization metadata

Phase 3 Outputs:
---------------
descriptor_results/successful_descriptors.csv - Quantum descriptors
descriptor_results/summary.json - Calculation statistics

Phase 4 Outputs:
---------------
des_features_final.csv - Final feature matrix (13 features)
descriptor_matching_log.csv - Component matching details
