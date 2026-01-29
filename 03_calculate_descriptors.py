#!/usr/bin/env python3
"""
COMPLETE DESCRIPTOR EXTRACTION SCRIPT
Extracts all available xTB descriptors for your DES components
"""

import os
import subprocess
import pandas as pd
import re
import numpy as np
from pathlib import Path

def extract_descriptors_from_log(content, molecule_name):
    """Extract all descriptors from xTB output"""
    
    descriptors = {
        'molecule': molecule_name,
        'status': 'success'
    }
    
    # 1. TOTAL ENERGY
    energy_match = re.search(r'TOTAL ENERGY\s+([-]?\d+\.\d+)\s+Eh', content)
    if energy_match:
        descriptors['total_energy_eh'] = float(energy_match.group(1))
    
    # 2. HOMO-LUMO GAP
    gap_match = re.search(r'HOMO-LUMO GAP\s+([-]?\d+\.\d+)\s+eV', content)
    if gap_match:
        descriptors['homo_lumo_gap_ev'] = float(gap_match.group(1))
    
    # 3. HOMO and LUMO energies (from orbital listing)
    # Look for lines with (HOMO) and (LUMO) markers
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '(HOMO)' in line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    descriptors['homo_energy_ev'] = float(parts[-2])
                except:
                    pass
        elif '(LUMO)' in line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    descriptors['lumo_energy_ev'] = float(parts[-2])
                except:
                    pass
    
    # 4. Solvation energy
    gsolv_match = re.search(r'Gsolv\s+([-]?\d+\.\d+)\s+Eh', content)
    if gsolv_match:
        descriptors['solvation_energy_eh'] = float(gsolv_match.group(1))
    
    # 5. Dipole moment
    dipole_match = re.search(r'q only:\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)', content)
    if dipole_match:
        descriptors['dipole_x'] = float(dipole_match.group(1))
        descriptors['dipole_y'] = float(dipole_match.group(2))
        descriptors['dipole_z'] = float(dipole_match.group(3))
        descriptors['dipole_total'] = float(dipole_match.group(4))
    
    # 6. Polarizability (if available)
    polar_match = re.search(r'Mol\. a\(0\) /au\s*:\s*(\d+\.\d+)', content)
    if polar_match:
        descriptors['polarizability_au'] = float(polar_match.group(1))
    
    # 7. Dispersion energy
    disp_match = re.search(r'dispersion\s+([-]?\d+\.\d+)\s+Eh', content)
    if disp_match:
        descriptors['dispersion_energy_eh'] = float(disp_match.group(1))
    
    # 8. Gradient norm (convergence quality)
    grad_match = re.search(r'GRADIENT NORM\s+(\d+\.\d+)\s+Eh/a', content)
    if grad_match:
        descriptors['gradient_norm'] = float(grad_match.group(1))
    
    return descriptors

def calculate_derived_descriptors(descriptors):
    """Calculate derived descriptors"""
    derived = {}
    
    # Electronic hardness/softness (from HOMO-LUMO gap)
    if 'homo_lumo_gap_ev' in descriptors:
        gap_ev = descriptors['homo_lumo_gap_ev']
        gap_hartree = gap_ev / 27.2114  # Convert eV to Hartree
        if gap_hartree > 0:
            derived['chemical_hardness'] = gap_hartree / 2
            derived['chemical_softness'] = 1 / derived['chemical_hardness']
    
    # Electronegativity (Mulliken)
    if 'homo_energy_ev' in descriptors and 'lumo_energy_ev' in descriptors:
        homo_ev = descriptors['homo_energy_ev']
        lumo_ev = descriptors['lumo_energy_ev']
        derived['mulliken_electronegativity_ev'] = -(homo_ev + lumo_ev) / 2
    
    # Global electrophilicity index
    if 'mulliken_electronegativity_ev' in derived and 'chemical_hardness' in derived:
        mu = derived['mulliken_electronegativity_ev'] / 27.2114  # eV to Hartree
        eta = derived['chemical_hardness']
        derived['electrophilicity_index'] = (mu**2) / (2 * eta)
    
    return derived

def run_xtb_for_molecule(xyz_file, charge=0):
    """Run xTB calculation for a single molecule"""
    molecule = os.path.basename(xyz_file).replace('_opt.xyz', '')
    
    print(f"\n?? Processing {molecule} (charge={charge})...")
    
    # Command that works with your xTB
    cmd = f"xtb {xyz_file} --gfn2 --chrg {charge} --alpb water --pop --vfukui"
    
    temp_log = f"temp_{molecule}.log"
    
    try:
        # Run xTB
        with open(temp_log, 'w') as f:
            subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, 
                          timeout=600, text=True)
        
        # Read and parse output
        with open(temp_log, 'r') as f:
            content = f.read()
        
        # Extract descriptors
        descriptors = extract_descriptors_from_log(content, molecule)
        
        # Calculate derived descriptors
        derived = calculate_derived_descriptors(descriptors)
        
        # Combine all descriptors
        result = {**descriptors, **derived}
        result['charge'] = charge
        
        # Cleanup
        os.remove(temp_log)
        
        # Print key results
        if 'total_energy_eh' in result:
            print(f"   ? Energy: {result['total_energy_eh']:.4f} Eh")
        if 'homo_lumo_gap_ev' in result:
            print(f"   ? HOMO-LUMO gap: {result['homo_lumo_gap_ev']:.2f} eV")
        if 'dipole_total' in result:
            print(f"   ? Dipole: {result['dipole_total']:.2f} Debye")
        
        return result
        
    except Exception as e:
        print(f"   ? Error: {str(e)[:50]}")
        if os.path.exists(temp_log):
            os.remove(temp_log)
        return {'molecule': molecule, 'status': 'failed', 'error': str(e)}

def main():
    """Main pipeline"""
    print("=" * 70)
    print("DES COMPONENT DESCRIPTOR CALCULATION PIPELINE")
    print("=" * 70)
    
    # Directory with optimized structures
    opt_dir = '../02_geometry_optimization/optimized_structures'
    
    if not os.path.exists(opt_dir):
        print(f"? Directory not found: {opt_dir}")
        return
    
    # Get all optimized molecules
    xyz_files = sorted([f for f in os.listdir(opt_dir) if f.endswith('_opt.xyz')])
    
    if not xyz_files:
        print("? No optimized structures found")
        return
    
    print(f"Found {len(xyz_files)} optimized molecules")
    
    # Molecule charges (from your dataset)
    molecule_charges = {
        'choline': 1,
        'chloride': -1,
        'glycerol': 0,
        'ethylene_glycol': 0,
        'urea': 0,
        'acetic_acid': 0,
        'oxalic_acid': 0,
        'lactic_acid': 0,
        'malonic_acid': 0,
        'glucose': 0,
        'sorbitol': 0,
        'betaine': 0,
        'acetate_ion': -1,
        'acetate': -1  # Handle both names
    }
    
    # Create output directory
    output_dir = 'descriptor_results'
    Path(output_dir).mkdir(exist_ok=True)
    
    results = []
    
    print(f"\n{'='*50}")
    print("STARTING CALCULATIONS")
    print(f"{'='*50}")
    
    # Process each molecule
    for i, xyz_file in enumerate(xyz_files, 1):
        molecule_name = xyz_file.replace('_opt.xyz', '')
        full_path = os.path.join(opt_dir, xyz_file)
        
        # Get charge
        charge = molecule_charges.get(molecule_name, 0)
        
        # Run calculation
        descriptors = run_xtb_for_molecule(full_path, charge)
        results.append(descriptors)
        
        # Save intermediate results every 3 molecules
        if i % 3 == 0:
            df_temp = pd.DataFrame([r for r in results if r])
            df_temp.to_csv(f'{output_dir}/intermediate_results.csv', index=False)
    
    # Filter out failed calculations
    successful = [r for r in results if r.get('status') == 'success']
    
    if successful:
        # Create final DataFrames
        df_all = pd.DataFrame(results)
        df_success = pd.DataFrame(successful)
        
        # Save results
        df_all.to_csv(f'{output_dir}/all_results.csv', index=False)
        df_success.to_csv(f'{output_dir}/successful_descriptors.csv', index=False)
        
        # Create summary statistics
        summary = {
            'total_molecules': len(xyz_files),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'success_rate': f"{(len(successful)/len(xyz_files))*100:.1f}%",
            'available_descriptors': list(df_success.columns) if not df_success.empty else []
        }
        
        with open(f'{output_dir}/summary.json', 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n{'='*70}")
        print("? PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Total molecules processed: {len(xyz_files)}")
        print(f"Successful calculations: {len(successful)}")
        print(f"Success rate: {summary['success_rate']}")
        
        print(f"\n?? Available descriptors ({len(df_success.columns)} total):")
        for col in ['total_energy_eh', 'homo_lumo_gap_ev', 'dipole_total', 
                   'solvation_energy_eh', 'chemical_hardness']:
            if col in df_success.columns:
                mean_val = df_success[col].mean()
                print(f"   {col:25}: {mean_val:.4f} (average)")
        
        print(f"\n?? Output files:")
        print(f"   {output_dir}/successful_descriptors.csv - Complete descriptor table")
        print(f"   {output_dir}/all_results.csv - All results (including failed)")
        print(f"   {output_dir}/summary.json - Summary statistics")
        
    else:
        print(f"\n? No successful calculations")
        print("Check individual error messages above")

if __name__ == "__main__":
    main()
