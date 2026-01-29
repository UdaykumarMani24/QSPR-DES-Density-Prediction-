import os
import subprocess
import pandas as pd
import json
from datetime import datetime
import shutil
import time

def optimize_molecules():
    """Optimize molecules using OpenBabel for geometry and xTB for optimization"""
    
    print("=" * 80)
    print("MOLECULE OPTIMIZATION PIPELINE")
    print("=" * 80)
    
    # Read molecules from file
    molecules = read_molecules_file('../01_data_preparation/unique_molecules_CORRECTED.txt')
    if not molecules:
        print("? No molecules loaded. Check file format.")
        return
    
    print(f"? Loaded {len(molecules)} molecules for optimization")
    print("\n" + "-" * 80)
    
    # Create output directories
    os.makedirs('optimized_structures', exist_ok=True)
    os.makedirs('xtb_logs', exist_ok=True)
    os.makedirs('initial_geometries', exist_ok=True)
    
    results = []
    
    # Process each molecule
    for i, mol in enumerate(molecules, 1):
        print(f"\n[{i:2d}/{len(molecules)}] ?? {mol['name']} ({mol['formula']})")
        print(f"   SMILES: {mol['smiles']}")
        print(f"   Charge: {mol['charge']}, Atoms: {mol.get('natoms', 'N/A')}")
        
        # Step 1: Generate initial geometry
        print("   ?? Generating 3D geometry...", end="")
        init_xyz = generate_initial_geometry(mol)
        
        if not init_xyz:
            print(" ? FAILED")
            results.append(create_failure_result(mol, "Geometry generation failed"))
            continue
        
        print(" ?")
        
        # Step 2: Run xTB optimization
        print("   ??  Running xTB optimization...", end="")
        success, energy, opt_xyz = run_xtb_optimization(init_xyz, mol)
        
        if not success:
            print(" ? FAILED")
            results.append(create_failure_result(mol, "xTB optimization failed"))
            cleanup_temp_files(mol['name'])
            continue
        
        print(f" ? Energy: {energy:.6f} Eh")
        
        # Step 3: Save results
        save_optimized_structure(mol, opt_xyz, energy)
        
        results.append(create_success_result(mol, energy, opt_xyz))
        
        # Step 4: Cleanup
        cleanup_temp_files(mol['name'])
        
        # Small delay between runs
        time.sleep(0.5)
    
    # Save comprehensive results
    save_final_results(results, molecules)
    
    print("\n" + "=" * 80)
    print("?? OPTIMIZATION COMPLETE!")
    print("=" * 80)

def read_molecules_file(filename):
    """Read molecules from file with flexible parsing"""
    molecules = []
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Skip header if present
        start_idx = 0
        if lines[0].startswith('#') or 'SMILES' in lines[0]:
            start_idx = 1
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Flexible parsing - handle both tabs and spaces
            parts = line.split('\t') if '\t' in line else line.split()
            
            if len(parts) >= 7:
                try:
                    mol = {
                        'smiles': parts[0],
                        'name': parts[1],
                        'charge': int(parts[2]),
                        'type': parts[3],
                        'formula': parts[4],
                        'mw': float(parts[5]),
                        'natoms': int(parts[6])
                    }
                    molecules.append(mol)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line[:50]}...")
                    continue
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return []
    
    return molecules

def generate_initial_geometry(mol):
    """Generate 3D geometry using OpenBabel"""
    name = mol['name'].replace(' ', '_').replace('(', '').replace(')', '')
    smiles = mol['smiles']
    
    # Create temporary files
    smi_file = f"temp_{name}.smi"
    xyz_file = f"temp_{name}.xyz"
    
    # Write SMILES file
    with open(smi_file, 'w') as f:
        f.write(f"{smiles}\t{name}\n")
    
    # Try OpenBabel with different options
    obabel_cmds = [
        f'obabel -ismi {smi_file} -oxyz --gen3d --minimize --ff MMFF94 -O {xyz_file}',
        f'obabel -ismi {smi_file} -oxyz --gen3d --minimize -O {xyz_file}',
        f'obabel -ismi {smi_file} -oxyz --gen3d -O {xyz_file}'
    ]
    
    for cmd in obabel_cmds:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if os.path.exists(xyz_file):
            break
    
    if not os.path.exists(xyz_file):
        os.remove(smi_file)
        return None
    
    # Read and fix XYZ file
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    # Ensure proper format
    if len(lines) >= 2:
        atom_count = len(lines) - 2
        lines[0] = f"{atom_count}\n"
        lines[1] = f"{name} charge={mol['charge']}\n"
        
        with open(xyz_file, 'w') as f:
            f.writelines(lines)
    
    # Save initial geometry for reference
    init_save = f"initial_geometries/{name}_init.xyz"
    shutil.copy(xyz_file, init_save)
    
    # Cleanup
    os.remove(smi_file)
    
    return xyz_file

def run_xtb_optimization(xyz_file, mol):
    """Run xTB optimization"""
    name = mol['name'].replace(' ', '_').replace('(', '').replace(')', '')
    charge = mol['charge']
    
    # Create xTB command with robust settings
    xtb_cmd = f'xtb {xyz_file} --opt --gfn2 --chrg {charge} '
    xtb_cmd += f'--cycles 300 --etemp 3000 --gfn2 '
    xtb_cmd += f'--grad 0.01 --energy 0.0001 '
    xtb_cmd += f'> {name}_xtb.log 2>&1'
    
    try:
        # Run xTB with timeout
        result = subprocess.run(
            xtb_cmd, shell=True,
            timeout=600,  # 10 minutes timeout
            capture_output=False
        )
        
        # Check if optimization succeeded
        if os.path.exists('xtbopt.xyz'):
            # Read optimized structure
            with open('xtbopt.xyz', 'r') as f:
                opt_structure = f.read()
            
            # Parse energy from log
            energy = parse_energy_from_log(f'{name}_xtb.log')
            
            # Save log
            log_dest = f"xtb_logs/{name}_xtb.log"
            if os.path.exists(f'{name}_xtb.log'):
                shutil.move(f'{name}_xtb.log', log_dest)
            
            return True, energy, opt_structure
        
        return False, None, None
        
    except subprocess.TimeoutExpired:
        print(f"Timeout for {name}")
        return False, None, None
    except Exception as e:
        print(f"Error for {name}: {str(e)[:100]}")
        return False, None, None

def parse_energy_from_log(log_file):
    """Parse energy from xTB log file"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for energy in log
        for line in content.split('\n'):
            if 'TOTAL ENERGY' in line:
                try:
                    return float(line.split()[3])
                except:
                    pass
        
        # Alternative patterns
        for line in content.split('\n'):
            if 'total energy' in line.lower() and 'Eh' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'Eh' in part and i > 0:
                        try:
                            return float(parts[i-1])
                        except:
                            pass
    
    except:
        pass
    
    return None

def save_optimized_structure(mol, opt_structure, energy):
    """Save optimized structure to file"""
    name = mol['name'].replace(' ', '_').replace('(', '').replace(')', '')
    
    filename = f"optimized_structures/{name}_opt.xyz"
    
    # Add energy to comment line
    lines = opt_structure.split('\n')
    if len(lines) >= 2:
        lines[1] = f"{name} E={energy:.8f} Eh charge={mol['charge']}"
        
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
    
    return filename

def create_success_result(mol, energy, opt_structure):
    """Create result entry for successful optimization"""
    return {
        'name': mol['name'],
        'smiles': mol['smiles'],
        'formula': mol['formula'],
        'charge': mol['charge'],
        'type': mol['type'],
        'success': True,
        'energy_eh': energy,
        'energy_kcal': energy * 627.509 if energy else None,
        'file': f"optimized_structures/{mol['name'].replace(' ', '_')}_opt.xyz",
        'error': None,
        'timestamp': datetime.now().isoformat()
    }

def create_failure_result(mol, error):
    """Create result entry for failed optimization"""
    return {
        'name': mol['name'],
        'smiles': mol['smiles'],
        'formula': mol['formula'],
        'charge': mol['charge'],
        'type': mol['type'],
        'success': False,
        'energy_eh': None,
        'energy_kcal': None,
        'file': None,
        'error': error,
        'timestamp': datetime.now().isoformat()
    }

def cleanup_temp_files(name):
    """Clean up temporary files"""
    name_clean = name.replace(' ', '_').replace('(', '').replace(')', '')
    
    files_to_remove = [
        f'temp_{name_clean}.smi',
        f'temp_{name_clean}.xyz',
        f'{name_clean}_xtb.log',
        'xtbopt.xyz',
        'xtbrestart',
        'charges',
        'wbo',
        'xtbopt.log',
        'gfnff_topo',
        '.xtboptok',
        'xtbtopo.mol',
        'control.inp'
    ]
    
    for file in files_to_remove:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass

def save_final_results(results, molecules):
    """Save all results to files"""
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv('optimization_results.csv', index=False)
    
    # Save summary statistics
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    summary = {
        'date': datetime.now().isoformat(),
        'total_molecules': total,
        'successful_optimizations': successful,
        'success_rate': f"{(successful/total)*100:.1f}%" if total > 0 else "0%",
        'software_versions': {
            'xtb': get_xtb_version(),
            'openbabel': get_openbabel_version()
        },
        'parameters': {
            'method': 'GFN2-xTB',
            'optimization': 'ANC optimizer',
            'convergence': 'gradient < 0.01 Eh/a0'
        }
    }
    
    with open('optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("?? OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Total molecules processed: {total}")
    print(f"Successful optimizations: {successful}")
    print(f"Success rate: {summary['success_rate']}")
    
    if successful > 0:
        print(f"\n? Optimized structures saved to: optimized_structures/")
        print(f"?? Detailed results: optimization_results.csv")
        print(f"?? Summary: optimization_summary.json")
    else:
        print("\n? No successful optimizations.")
        print("Check xtb_logs/ for error details.")

def get_xtb_version():
    """Get xTB version"""
    try:
        result = subprocess.run('xtb --version', shell=True, capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'xtb version' in line:
                return line.strip()
    except:
        pass
    return "Unknown"

def get_openbabel_version():
    """Get OpenBabel version"""
    try:
        result = subprocess.run('obabel --version', shell=True, capture_output=True, text=True)
        return result.stdout.split('\n')[0].strip()
    except:
        pass
    return "Unknown"

if __name__ == "__main__":
    # Check dependencies
    print("?? Checking dependencies...")
    for cmd, name in [('xtb', 'xTB'), ('obabel', 'OpenBabel')]:
        result = subprocess.run(f'which {cmd}', shell=True, capture_output=True)
        if result.returncode == 0:
            print(f"? {name} found")
        else:
            print(f"? {name} NOT found")
    
    print("\n" + "=" * 80)
    
    # Run optimization
    optimize_molecules()
