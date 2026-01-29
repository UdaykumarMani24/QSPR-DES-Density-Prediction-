#!/usr/bin/env python3
"""
FINAL FEATURE ENGINEERING - Fixed matching by SMILES and names
"""

import pandas as pd
import numpy as np
import re
import os

def safe_read_csv(filepath):
    """Safely read CSV file, handling parsing errors"""
    try:
        # Try reading with error handling
        df = pd.read_csv(filepath, dtype=str, keep_default_na=False, engine='python', on_bad_lines='warn')
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        print("Attempting manual parsing...")
        
        # Manual parsing as fallback
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            raise ValueError(f"CSV file {filepath} is empty")
        
        headers = lines[0].split(',')
        headers = [h.strip() for h in headers]
        
        data = []
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) != len(headers):
                # Fix by padding or truncating
                if len(parts) > len(headers):
                    parts = parts[:len(headers)]
                else:
                    parts = parts + [''] * (len(headers) - len(parts))
            data.append(parts)
        
        df = pd.DataFrame(data, columns=headers)
        return df

def load_molecule_mapping():
    """Load molecule information from Phase 1 to create SMILES-name mapping"""
    mapping = {}
    
    # Try to load from unique molecules file
    molecules_file = '../01_data_preparation/unique_molecules_CORRECTED.txt'
    if os.path.exists(molecules_file):
        print(f"Loading molecule mapping from {molecules_file}")
        with open(molecules_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header line
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 6:  # SMILES, Name, Charge, Type, Formula, MW
                smiles = parts[0].strip()
                name = parts[1].strip()
                mol_type = parts[3].strip() if len(parts) > 3 else ''
                
                if smiles and name:
                    mapping[smiles] = {
                        'name': name,
                        'type': mol_type,
                        'clean_name': name.lower().replace(' ', '_').replace(',', '').replace('-', '_')
                    }
    
    print(f"Loaded {len(mapping)} molecule mappings")
    return mapping

def extract_components_from_des(des_row, molecule_mapping):
    """Extract HBA and HBD information from DES row"""
    des_name = des_row['Des_name']
    hba_smiles = str(des_row['hba_smiles']).strip()
    hbd_smiles = str(des_row['hbd_smiles']).strip()
    
    # Handle salt notation (replace '.' with '|' if needed)
    if '.' in hba_smiles and '|' not in hba_smiles and '[' in hba_smiles and ']' in hba_smiles:
        hba_smiles = re.sub(r'(\[[^\]]+\])\.(\[[^\]]+\])', r'\1|\2', hba_smiles)
    
    # Get HBA component (first part if it's a salt)
    hba_primary = hba_smiles.split('|')[0] if '|' in hba_smiles else hba_smiles
    hbd_primary = hbd_smiles
    
    # Try to get names from mapping
    hba_name = None
    hbd_name = None
    
    if hba_primary in molecule_mapping:
        hba_name = molecule_mapping[hba_primary]['clean_name']
    else:
        # Try to find by partial match
        for smiles, info in molecule_mapping.items():
            if smiles in hba_primary or hba_primary in smiles:
                hba_name = info['clean_name']
                break
    
    if hbd_primary in molecule_mapping:
        hbd_name = molecule_mapping[hbd_primary]['clean_name']
    else:
        # Try to find by partial match
        for smiles, info in molecule_mapping.items():
            if smiles in hbd_primary or hbd_primary in smiles:
                hbd_name = info['clean_name']
                break
    
    # If still no name, create a simple name from SMILES
    if not hba_name:
        hba_name = f"hba_{hash(hba_primary) % 10000:04d}"
    
    if not hbd_name:
        hbd_name = f"hbd_{hash(hbd_primary) % 10000:04d}"
    
    return {
        'des_name': des_name,
        'hba_smiles': hba_smiles,
        'hbd_smiles': hbd_smiles,
        'hba_primary': hba_primary,
        'hbd_primary': hbd_primary,
        'hba_name': hba_name,
        'hbd_name': hbd_name,
        'hba_moles': float(des_row['hba_moles']) if str(des_row['hba_moles']).strip() else 1.0,
        'hbd_moles': float(des_row['hbd_moles']) if str(des_row['hbd_moles']).strip() else 1.0,
        'molar_ratio': des_row['molar_ratio']
    }

def normalize_descriptor_name(name):
    """Normalize descriptor name for matching"""
    if not isinstance(name, str):
        return ""
    
    name = name.strip().lower()
    
    # Remove special characters and spaces
    name = re.sub(r'[^a-z0-9_]', '', name)
    
    # Common replacements
    replacements = {
        '12butanediol': '1_2_butanediol',
        '16hexanediol': '1_6_hexanediol',
        'ethylene_glycol': 'ethylene_glycol',
        'lactic_acid': 'lactic_acid',
        'citric_acid': 'citric_acid',
        'oxalic_acid': 'oxalic_acid',
        'malic_acid': 'malic_acid',
        'tartaric_acid': 'tartaric_acid',
        'imidazole': 'imidazole',
        'glucose': 'glucose',
        'fructose': 'fructose',
        'sucrose': 'sucrose',
        'glycerol': 'glycerol',
    }
    
    for old, new in replacements.items():
        if name == old:
            name = new
    
    return name

def find_matching_descriptor(component_name, desc_df, molecule_name_col='molecule'):
    """Find matching descriptor for a component name"""
    if component_name is None:
        return None
    
    # Normalize the component name
    normalized_component = normalize_descriptor_name(component_name)
    
    # Try exact match
    for idx, row in desc_df.iterrows():
        desc_name = str(row[molecule_name_col]).strip()
        normalized_desc = normalize_descriptor_name(desc_name)
        
        if normalized_component == normalized_desc:
            return row
    
    # Try partial match
    for idx, row in desc_df.iterrows():
        desc_name = str(row[molecule_name_col]).strip().lower()
        component_lower = component_name.lower()
        
        if (component_lower in desc_name or 
            desc_name in component_lower or
            normalized_component in desc_name or
            desc_name in normalized_component):
            return row
    
    # Try removing underscores and numbers
    simple_component = re.sub(r'[_\d]', '', component_name.lower())
    for idx, row in desc_df.iterrows():
        desc_name = str(row[molecule_name_col]).strip().lower()
        simple_desc = re.sub(r'[_\d]', '', desc_name)
        
        if simple_component == simple_desc:
            return row
    
    return None

def create_descriptor_lookup(desc_df):
    """Create multiple lookup dictionaries for descriptor matching"""
    lookups = {
        'exact': {},      # Exact name matches
        'normalized': {}, # Normalized name matches
        'partial': {}     # Partial name matches
    }
    
    for idx, row in desc_df.iterrows():
        molecule_name = str(row['molecule']).strip()
        
        # Exact match
        lookups['exact'][molecule_name] = row
        
        # Normalized match
        normalized = normalize_descriptor_name(molecule_name)
        lookups['normalized'][normalized] = row
        
        # Partial matches (key parts)
        name_parts = molecule_name.lower().replace('_', ' ').split()
        for part in name_parts:
            if len(part) > 3:  # Only meaningful parts
                lookups['partial'][part] = row
    
    return lookups

def main():
    print("=" * 70)
    print("FEATURE ENGINEERING - Fixed Version")
    print("=" * 70)
    
    try:
        # Load data
        print("\n1. Loading data...")
        df_des = safe_read_csv('../01_data_preparation/cleaned_dataset_CORRECTED.csv')
        df_desc = safe_read_csv('descriptor_results/successful_descriptors.csv')
        
        print(f"   DES formulations: {len(df_des)}")
        print(f"   Available descriptors: {len(df_desc)}")
        
        # Load molecule mapping from Phase 1
        print("\n2. Loading molecule mapping...")
        molecule_mapping = load_molecule_mapping()
        
        # Show sample descriptor names
        print(f"\n3. Sample descriptor names (first 15):")
        desc_names = df_desc['molecule'].unique()
        for i, name in enumerate(desc_names[:15]):
            print(f"   {i+1:2d}. {name}")
        
        # Create descriptor lookup
        print("\n4. Creating descriptor lookup...")
        lookups = create_descriptor_lookup(df_desc)
        print(f"   Exact matches: {len(lookups['exact'])}")
        print(f"   Normalized matches: {len(lookups['normalized'])}")
        print(f"   Partial matches: {len(lookups['partial'])}")
        
        # Process each DES
        print("\n5. Processing DES formulations...")
        all_features = []
        match_log = []
        
        successful_matches = 0
        
        for idx, row in df_des.iterrows():
            # Extract component information
            components = extract_components_from_des(row, molecule_mapping)
            
            # Find matching descriptors
            hba_desc = find_matching_descriptor(components['hba_name'], df_desc)
            hbd_desc = find_matching_descriptor(components['hbd_name'], df_desc)
            
            # Log matching info
            match_info = {
                'des_name': components['des_name'],
                'hba_primary': components['hba_primary'][:50],
                'hbd_primary': components['hbd_primary'][:50],
                'hba_name': components['hba_name'],
                'hbd_name': components['hbd_name'],
                'hba_found': hba_desc is not None,
                'hbd_found': hbd_desc is not None,
                'hba_match': hba_desc['molecule'] if hba_desc is not None else 'NO_MATCH',
                'hbd_match': hbd_desc['molecule'] if hbd_desc is not None else 'NO_MATCH',
            }
            match_log.append(match_info)
            
            # Calculate features if we found descriptors
            if hba_desc is not None or hbd_desc is not None:
                hba_moles = components['hba_moles']
                hbd_moles = components['hbd_moles']
                total_moles = hba_moles + hbd_moles
                
                if total_moles > 0:
                    hba_fraction = hba_moles / total_moles
                    hbd_fraction = hbd_moles / total_moles
                    
                    # Start with basic features
                    features = {
                        'des_name': components['des_name'],
                        'hba_name': components['hba_name'],
                        'hbd_name': components['hbd_name'],
                        'hba_smiles': components['hba_primary'][:100],  # Truncate if too long
                        'hbd_smiles': components['hbd_primary'],
                        'hba_moles': hba_moles,
                        'hbd_moles': hbd_moles,
                        'total_moles': total_moles,
                        'hbd_hba_ratio': hbd_moles / hba_moles if hba_moles > 0 else np.nan,
                        'hba_found': hba_desc is not None,
                        'hbd_found': hbd_desc is not None,
                    }
                    
                    # List of numeric descriptor columns to use
                    numeric_cols = [
                        'total_energy_eh', 'homo_lumo_gap_ev', 'homo_energy_ev',
                        'lumo_energy_ev', 'solvation_energy_eh', 'dipole_total',
                        'dispersion_energy_eh', 'chemical_hardness', 'chemical_softness',
                        'mulliken_electronegativity_ev', 'electrophilicity_index'
                    ]
                    
                    # Extract descriptor values
                    for col in numeric_cols:
                        hba_val = None
                        hbd_val = None
                        
                        if hba_desc is not None and col in hba_desc:
                            try:
                                hba_val = float(hba_desc[col])
                            except:
                                hba_val = np.nan
                        
                        if hbd_desc is not None and col in hbd_desc:
                            try:
                                hbd_val = float(hbd_desc[col])
                            except:
                                hbd_val = np.nan
                        
                        # Store individual values
                        if hba_val is not None and not np.isnan(hba_val):
                            features[f'hba_{col}'] = hba_val
                        
                        if hbd_val is not None and not np.isnan(hbd_val):
                            features[f'hbd_{col}'] = hbd_val
                        
                        # Calculate weighted average if both values exist
                        if (hba_val is not None and not np.isnan(hba_val) and 
                            hbd_val is not None and not np.isnan(hbd_val)):
                            features[f'weighted_{col}'] = hba_val * hba_fraction + hbd_val * hbd_fraction
                            features[f'{col}_difference'] = abs(hba_val - hbd_val)
                            if hbd_val != 0:
                                features[f'{col}_ratio'] = hba_val / hbd_val
                        elif hba_val is not None and not np.isnan(hba_val):
                            features[f'weighted_{col}'] = hba_val
                        elif hbd_val is not None and not np.isnan(hbd_val):
                            features[f'weighted_{col}'] = hbd_val
                    
                    all_features.append(features)
                    successful_matches += 1
                    
                    status = []
                    if hba_desc is not None: status.append("HBA")
                    if hbd_desc is not None: status.append("HBD")
                    print(f"   ? {components['des_name'][:50]:50} -> Found {', '.join(status)}")
                else:
                    print(f"   ? {components['des_name'][:50]:50} -> Zero total moles")
            else:
                print(f"   ? {components['des_name'][:50]:50} -> No descriptors found")
        
        # Save results
        print("\n6. Saving results...")
        if all_features:
            df_features = pd.DataFrame(all_features)
            
            # Save to CSV
            output_file = 'des_features_final.csv'
            df_features.to_csv(output_file, index=False)
            
            # Save match log
            df_match_log = pd.DataFrame(match_log)
            df_match_log.to_csv('descriptor_matching_log.csv', index=False)
            
            print(f"\n{'='*70}")
            print("FEATURE ENGINEERING COMPLETE!")
            print(f"{'='*70}")
            print(f"Successful matches: {successful_matches}/{len(df_des)}")
            print(f"Features generated: {len(df_features.columns)}")
            print(f"Rows in output: {len(df_features)}")
            print(f"\nOutput files:")
            print(f"  - {output_file}")
            print(f"  - descriptor_matching_log.csv")
            
            # Show statistics
            hba_found = sum(1 for f in all_features if f['hba_found'])
            hbd_found = sum(1 for f in all_features if f['hbd_found'])
            both_found = sum(1 for f in all_features if f['hba_found'] and f['hbd_found'])
            
            print(f"\nMatching statistics:")
            print(f"  HBA found: {hba_found}/{len(all_features)} ({hba_found/len(all_features)*100:.1f}%)")
            print(f"  HBD found: {hbd_found}/{len(all_features)} ({hbd_found/len(all_features)*100:.1f}%)")
            print(f"  Both found: {both_found}/{len(all_features)} ({both_found/len(all_features)*100:.1f}%)")
            
            # Show first few features
            print(f"\nFirst successful DES features:")
            if all_features:
                first = all_features[0]
                print(f"  DES: {first['des_name']}")
                print(f"  HBA: {first['hba_name']}")
                print(f"  HBD: {first['hbd_name']}")
                print(f"  Ratio: {first.get('hbd_hba_ratio', 'N/A')}")
                
                # Show some key features
                key_features = ['weighted_homo_lumo_gap_ev', 'homo_lumo_gap_ev_difference', 
                               'weighted_mulliken_electronegativity_ev']
                for key in key_features:
                    if key in first:
                        print(f"  {key}: {first[key]:.4f}")
            
            return df_features
        else:
            print("\nNo features generated!")
            
            # Debug information
            print("\nDebug information:")
            print(f"DES data shape: {df_des.shape}")
            print(f"Descriptor data shape: {df_desc.shape}")
            print(f"\nDescriptor columns: {df_desc.columns.tolist()}")
            
            return None
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    df = main()
    
    if df is not None:
        print(f"\n{'='*70}")
        print("NEXT STEPS:")
        print(f"{'='*70}")
        print("1. Check des_features_final.csv for all features")
        print("2. Check descriptor_matching_log.csv for matching details")
        print("3. The features are now ready for QSPR modeling")
        
        # Quick verification
        print("\nQuick verification:")
        print(f"Total DES processed: {len(df)}")
        print(f"Number of features: {len(df.columns)}")
        print("\nFirst 3 DES entries:")
        print(df[['des_name', 'hba_name', 'hbd_name', 'hbd_hba_ratio']].head(3).to_string())
