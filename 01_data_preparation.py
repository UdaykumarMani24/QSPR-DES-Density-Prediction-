import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import re
import json
from datetime import datetime
import warnings
from collections import defaultdict
import csv
warnings.filterwarnings('ignore')

class DESDatasetPreparer:
    def __init__(self, dataset_path='raw_dataset.csv'):
        self.dataset_path = dataset_path
        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'software_versions': {
                'rdkit': '2023.03.1',
                'pandas': pd.__version__,
                'numpy': np.__version__
            }
        }
    
    def load_and_validate_dataset(self):
        """COMPATIBLE: Load dataset with same behavior as original"""
        print("=" * 70)
        print("SCIENTIFIC DATASET PREPARATION (COMPATIBLE VERSION)")
        print("=" * 70)
        
        try:
            # Try reading with pandas first using python engine (more tolerant)
            print("Attempting to read CSV with pandas...")
            df = pd.read_csv(
                self.dataset_path, 
                dtype=str, 
                keep_default_na=False,
                engine='python',
                on_bad_lines='warn'
            )
            print("Successfully loaded with pandas")
            
        except Exception as e:
            print(f"Pandas failed to read CSV: {e}")
            print("\nAttempting manual CSV parsing...")
            
            # Manual parsing with proper CSV handling
            data = []
            with open(self.dataset_path, 'r') as f:
                # First, let's read the entire file
                content = f.read()
            
            # Split into lines
            lines = content.strip().split('\n')
            if not lines:
                raise ValueError("CSV file is empty")
            
            # Get headers from first line
            headers = lines[0].split(',')
            headers = [h.strip() for h in headers]
            print(f"Detected headers: {headers} ({len(headers)} columns)")
            
            # Process each data line
            for i, line in enumerate(lines[1:], start=2):
                line = line.strip()
                if not line:
                    continue
                    
                # Split by comma, but be careful with colons in DES names
                parts = line.split(',')
                
                # Check if we have the right number of columns
                if len(parts) != len(headers):
                    print(f"Warning: Line {i} has {len(parts)} columns, expected {len(headers)}")
                    print(f"  Line: {line[:100]}...")
                    
                    # Special handling for DES names with colons
                    # Some DES names like "X:Y" get split incorrectly
                    if len(parts) > len(headers):
                        # Check if first part ends with a colon and next part starts with a chemical
                        if parts[0].endswith(':') or (':' in parts[0] and not parts[0].endswith(':')):
                            # Try to combine first two parts as DES name
                            combined_name = f"{parts[0]},{parts[1]}"
                            # Check if this looks like a valid combination
                            if any(char.isdigit() for char in parts[1]) and ':' in parts[1]:
                                # parts[1] might be a ratio, don't combine
                                pass
                            else:
                                # Combine and adjust parts
                                new_parts = [combined_name] + parts[2:]
                                if len(new_parts) == len(headers):
                                    parts = new_parts
                                    print(f"  Fixed by combining DES name: {combined_name[:50]}...")
                    
                    # If still wrong, pad or truncate
                    if len(parts) > len(headers):
                        parts = parts[:len(headers)]
                        print(f"  Fixed by truncating to {len(headers)} columns")
                    elif len(parts) < len(headers):
                        parts = parts + [''] * (len(headers) - len(parts))
                        print(f"  Fixed by padding with empty strings")
                
                data.append(parts)
            
            df = pd.DataFrame(data, columns=headers)
            print(f"Manually loaded {len(df)} entries")
        
        print(f"\nLoaded dataset with {len(df)} entries")
        print(f"Columns: {df.columns.tolist()}")
        
        # Clean column names (strip whitespace)
        df.columns = [col.strip() for col in df.columns]
        
        # KEEP ORIGINAL COLUMN NAMES
        required_cols = ['Des_name', 'hba_smiles', 'hbd_smiles', 'hba_moles', 'hbd_moles', 'molar_ratio']
        
        # Show what columns we actually have
        print(f"\nCurrent columns in DataFrame: {df.columns.tolist()}")
        
        # Try to match columns case-insensitively
        column_mapping = {}
        actual_columns = df.columns.tolist()
        
        for req_col in required_cols:
            req_lower = req_col.lower()
            found = False
            for actual_col in actual_columns:
                if actual_col.lower() == req_lower:
                    column_mapping[actual_col] = req_col
                    found = True
                    break
            
            if not found:
                # Try partial matching
                for actual_col in actual_columns:
                    if req_lower in actual_col.lower() or actual_col.lower() in req_lower:
                        column_mapping[actual_col] = req_col
                        found = True
                        break
        
        if column_mapping:
            print(f"Renaming columns: {column_mapping}")
            df = df.rename(columns=column_mapping)
        
        # Check for missing columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"\nERROR: Missing required columns: {missing}")
            print(f"Current columns: {df.columns.tolist()}")
            print(f"Looking for: {required_cols}")
            
            # Try to identify which columns might match
            print("\nTrying to identify matching columns:")
            for req in missing:
                print(f"  Looking for '{req}':")
                for actual in df.columns:
                    similarity = sum(1 for a, b in zip(req.lower(), actual.lower()) if a == b) / max(len(req), len(actual))
                    if similarity > 0.6:
                        print(f"    Possible match: '{actual}' (similarity: {similarity:.2f})")
            
            raise ValueError(f"Missing required columns: {missing}")
        
        # Display sample of the data
        print("\n" + "=" * 70)
        print("DATASET LOADED SUCCESSFULLY")
        print("=" * 70)
        print(f"Shape: {df.shape}")
        print(f"Sample data (first 2 rows):")
        print(df.head(2).to_string())
        
        return df
    
    def get_molecule_info_CORRECTED(self, smiles, mol_type):
        """COMPATIBLE but with scientific rigor"""
        mol = Chem.MolFromSmiles(str(smiles))
        if not mol:
            return None
        
        # Add hydrogens for correct atom counts (scientific improvement)
        mol_with_h = Chem.AddHs(mol)
        
        # Get common name from your DES names
        common_names = {
            'C[N+](C)(C)CCO': 'choline',
            '[Cl-]': 'chloride',
            'C(CO)O': 'ethylene_glycol',
            'OCCCCCCO': '1,6-hexanediol',
            'CC(O)CCO': '1,2-butanediol',
            'C(C(CO)O)O': 'glycerol',
            'CC(C(=O)O)O': 'lactic_acid',
            'C(C(=O)O)C(CC(=O)O)(C(=O)O)O': 'citric_acid',
            'C(=O)(C(=O)O)O': 'oxalic_acid',
            'C(C(C(=O)O)O)C(=O)O': 'malic_acid',
            'C(C(C(=O)O)O)(C(=O)O)O': 'tartaric_acid',
            'CN1C=NC(=O)CC1': 'creatinine',
            'C1=CN=CN1': 'imidazole',
            'C(C1C(C(C(C(O1)O)O)O)O)O': 'glucose',
            'C1C(C(C(C(O1)(CO)O)O)O)O': 'fructose',
            'C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O': 'sucrose',
            'C(C1C(C(C(C(O1)OC2C(OC(C(C2O)O)O)CO)O)O)O)O': 'maltose',
        }
        
        name = common_names.get(str(smiles), Chem.MolToSmiles(mol, isomericSmiles=False)[:15])
        
        # Calculate properties WITH hydrogens (scientific improvement)
        charge = Chem.GetFormalCharge(mol_with_h)
        formula = rdMolDescriptors.CalcMolFormula(mol_with_h)
        mw = Descriptors.MolWt(mol_with_h)
        num_atoms = mol_with_h.GetNumAtoms()
        
        # ADDITIONAL properties for QSPR (scientific improvement)
        hbd_count = Descriptors.NumHDonors(mol_with_h)
        hba_count = Descriptors.NumHAcceptors(mol_with_h)
        logp = Descriptors.MolLogP(mol_with_h)
        tpsa = Descriptors.TPSA(mol_with_h)
        
        return {
            'name': name,
            'charge': charge,
            'type': mol_type,
            'formula': formula,
            'mw': mw,
            'num_atoms': num_atoms,
            'hbd_count': hbd_count,
            'hba_count': hba_count,
            'logp': logp,
            'tpsa': tpsa,
            'smiles': str(smiles),
            'rdkit_mol': mol_with_h
        }
    
    def extract_unique_molecules_CORRECTED(self, df):
        """COMPATIBLE output format but with better science"""
        molecule_info = {}
        
        print("\nExtracting molecules WITH hydrogens (scientifically correct)...")
        
        for _, row in df.iterrows():
            # Process HBA (maintain salt handling)
            hba_parts = str(row['hba_smiles']).split('|')
            for part in hba_parts:
                if part and part not in molecule_info:
                    info = self.get_molecule_info_CORRECTED(part, 'hba')
                    if info:
                        molecule_info[part] = info
            
            # Process HBD
            hbd = str(row['hbd_smiles'])
            if hbd and hbd not in molecule_info:
                info = self.get_molecule_info_CORRECTED(hbd, 'hbd')
                if info:
                    molecule_info[hbd] = info
        
        # Save in ORIGINAL format but with MORE data
        with open('unique_molecules_CORRECTED.txt', 'w') as f:
            f.write("# SMILES\tName\tCharge\tType\tFormula\tMW\tNumAtoms\tHBD_Count\tHBA_Count\tLogP\tTPSA\n")
            for smiles, info in molecule_info.items():
                f.write(f"{smiles}\t{info['name']}\t{info['charge']}\t"
                       f"{info['type']}\t{info['formula']}\t"
                       f"{info['mw']:.2f}\t{info['num_atoms']}\t"
                       f"{info['hbd_count']}\t{info['hba_count']}\t"
                       f"{info['logp']:.2f}\t{info['tpsa']:.2f}\n")
        
        print(f"\nExtracted {len(molecule_info)} unique molecules")
        return molecule_info
    
    def fix_smiles_issues(self, df):
        """COMPATIBLE SMILES fixing"""
        def fix_smiles(smiles):
            if pd.isna(smiles):
                return smiles
            s = str(smiles).strip()
            if '.' in s and '|' not in s and '[' in s and ']' in s:
                s = re.sub(r'(\[[^\]]+\])\.(\[[^\]]+\])', r'\1|\2', s)
            return s
        
        df['hba_smiles_fixed'] = df['hba_smiles'].apply(fix_smiles)
        df['hbd_smiles_fixed'] = df['hbd_smiles'].apply(fix_smiles)
        return df
    
    def run_full_preparation_CORRECTED(self):
        """MAIN FUNCTION - Creates SAME outputs as original"""
        self.df = self.load_and_validate_dataset()
        
        print("\nFixing SMILES issues...")
        self.df = self.fix_smiles_issues(self.df)
        
        # Save cleaned dataset (SAME filename as original)
        self.df.to_csv('cleaned_dataset_CORRECTED.csv', index=False)
        print("Saved cleaned_dataset_CORRECTED.csv")
        
        # Extract molecules (creates SAME output file)
        self.molecule_info = self.extract_unique_molecules_CORRECTED(self.df)
        
        # Save metadata (SAME filename)
        metadata = {
            'dataset_info': {
                'source': self.dataset_path,
                'preparation_date': datetime.now().isoformat(),
                'total_entries': len(self.df),
                'unique_molecules': len(self.molecule_info)
            },
            'note': 'Enhanced version with hydrogen atoms for correct properties',
            'software': self.metadata['software_versions'],
            'descriptors_included': ['HBD_Count', 'HBA_Count', 'LogP', 'TPSA']  # Added info
        }
        
        with open('dataset_metadata_CORRECTED.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\nMetadata saved to dataset_metadata_CORRECTED.json")
        
        # ADDITIONAL OUTPUTS for QSPR/optimization (optional)
        self.create_qspr_ready_outputs()
        
        print("\n" + "=" * 70)
        print("DATA PREPARATION COMPLETE (COMPATIBLE VERSION)")
        print("=" * 70)
        print("Output files generated:")
        print("1. cleaned_dataset_CORRECTED.csv - Cleaned dataset (COMPATIBLE)")
        print("2. unique_molecules_CORRECTED.txt - Molecule info with QSPR descriptors (ENHANCED)")
        print("3. dataset_metadata_CORRECTED.json - Metadata (COMPATIBLE)")
        print("\nADDITIONAL FILES FOR QSPR:")
        print("4. des_composition_qspr.csv - Ready for QSPR modeling")
        print("5. molecular_descriptors.csv - All calculated descriptors")
        
        return self.df, self.molecule_info
    
    def create_qspr_ready_outputs(self):
        """Create additional files for QSPR/optimization"""
        if not hasattr(self, 'molecule_info'):
            return
        
        # 1. DES composition table for QSPR
        qspr_data = []
        for idx, row in self.df.iterrows():
            entry = {
                'DES_ID': f"DES_{idx+1:04d}",
                'DES_name': row['Des_name'],
                'hba_smiles': row['hba_smiles'],
                'hbd_smiles': row['hbd_smiles'],
                'hba_moles': row['hba_moles'],
                'hbd_moles': row['hbd_moles'],
                'molar_ratio': row['molar_ratio']
            }
            
            # Add HBA properties
            hba_smiles = str(row['hba_smiles']).split('|')[0] if '|' in str(row['hba_smiles']) else str(row['hba_smiles'])
            if hba_smiles in self.molecule_info:
                hba_info = self.molecule_info[hba_smiles]
                entry.update({
                    'HBA_MW': hba_info['mw'],
                    'HBA_HBD': hba_info['hbd_count'],
                    'HBA_HBA': hba_info['hba_count'],
                    'HBA_LogP': hba_info['logp'],
                    'HBA_TPSA': hba_info['tpsa'],
                    'HBA_charge': hba_info['charge']
                })
            
            # Add HBD properties
            hbd_smiles = str(row['hbd_smiles'])
            if hbd_smiles in self.molecule_info:
                hbd_info = self.molecule_info[hbd_smiles]
                entry.update({
                    'HBD_MW': hbd_info['mw'],
                    'HBD_HBD': hbd_info['hbd_count'],
                    'HBD_HBA': hbd_info['hba_count'],
                    'HBD_LogP': hbd_info['logp'],
                    'HBD_TPSA': hbd_info['tpsa'],
                    'HBD_charge': hbd_info['charge']
                })
            
            qspr_data.append(entry)
        
        qspr_df = pd.DataFrame(qspr_data)
        qspr_df.to_csv('des_composition_qspr.csv', index=False)
        
        # 2. Molecular descriptors table
        descriptors_data = []
        for smiles, info in self.molecule_info.items():
            desc = {
                'SMILES': smiles,
                'Name': info['name'],
                'Type': info['type'],
                'Formula': info['formula'],
                'MW': info['mw'],
                'Charge': info['charge'],
                'Num_Atoms': info['num_atoms'],
                'HBD_Count': info['hbd_count'],
                'HBA_Count': info['hba_count'],
                'LogP': info['logp'],
                'TPSA': info['tpsa']
            }
            descriptors_data.append(desc)
        
        desc_df = pd.DataFrame(descriptors_data)
        desc_df.to_csv('molecular_descriptors.csv', index=False)
        
        print("\n? Created QSPR-ready files:")
        print("   - des_composition_qspr.csv: DES data with molecular properties")
        print("   - molecular_descriptors.csv: All calculated molecular descriptors")

if __name__ == "__main__":
    try:
        # Initialize with YOUR dataset
        preparer = DESDatasetPreparer('raw_dataset.csv')
        
        # Run the compatible version
        df, molecules = preparer.run_full_preparation_CORRECTED()
        
        # Show verification
        print("\n" + "=" * 70)
        print("VERIFICATION OF COMPATIBLE OUTPUTS:")
        print("=" * 70)
        
        import os
        expected_files = [
            'cleaned_dataset_CORRECTED.csv',
            'unique_molecules_CORRECTED.txt', 
            'dataset_metadata_CORRECTED.json',
            'des_composition_qspr.csv',
            'molecular_descriptors.csv'
        ]
        
        for file in expected_files:
            if os.path.exists(file):
                print(f"? {file} - EXISTS ({os.path.getsize(file)} bytes)")
            else:
                print(f"? {file} - MISSING")
                
    except Exception as e:
        print(f"\n? ERROR: {e}")
        import traceback
        traceback.print_exc()
