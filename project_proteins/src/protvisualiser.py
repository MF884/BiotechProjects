import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests
import io
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Atom:
    """Represents an atom in a protein structure"""
    atom_id: int
    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: int
    x: float
    y: float
    z: float
    element: str

class ProteinVisualizer:
    """A class for visualizing protein structures"""
    
    def __init__(self):
        self.atoms: List[Atom] = []
        self.protein_name = ""
        
        # Color scheme for different elements
        self.element_colors = {
            'C': '#909090',  # Carbon - gray
            'N': '#3050F8',  # Nitrogen - blue
            'O': '#FF0D0D',  # Oxygen - red
            'S': '#FFFF30',  # Sulfur - yellow
            'P': '#FF8000',  # Phosphorus - orange
            'H': '#FFFFFF',  # Hydrogen - white
            'FE': '#FFA500', # Iron - orange
            'ZN': '#7D80B0', # Zinc - blue-gray
            'MG': '#8AFF00', # Magnesium - green
            'CA': '#3DFF00', # Calcium - green
            'default': '#FF69B4'  # Pink for unknown elements
        }
        
        # Color scheme for different residue types
        self.residue_colors = {
            # Hydrophobic
            'ALA': '#C8C8C8', 'VAL': '#C8C8C8', 'LEU': '#C8C8C8', 
            'ILE': '#C8C8C8', 'MET': '#C8C8C8', 'PHE': '#C8C8C8',
            'TRP': '#C8C8C8', 'PRO': '#C8C8C8',
            # Polar
            'SER': '#00FF00', 'THR': '#00FF00', 'TYR': '#00FF00',
            'ASN': '#00FF00', 'GLN': '#00FF00', 'CYS': '#FFFF00',
            # Charged positive
            'ARG': '#0000FF', 'LYS': '#0000FF', 'HIS': '#0000FF',
            # Charged negative
            'ASP': '#FF0000', 'GLU': '#FF0000',
            # Special
            'GLY': '#FFFFFF',
            # Default
            'default': '#FF69B4'
        }
    
    def download_pdb(self, pdb_id: str) -> str:
        """Download PDB file from RCSB PDB database"""
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            print(f"Successfully downloaded PDB file for {pdb_id}")
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error downloading PDB file: {e}")
            return ""
    
    def parse_pdb_content(self, pdb_content: str) -> None:
        """Parse PDB file content and extract atomic coordinates"""
        self.atoms = []
        lines = pdb_content.strip().split('\n')
        
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    atom_id = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    residue_number = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    element = line[76:78].strip() if len(line) > 76 else atom_name[0]
                    
                    atom = Atom(atom_id, atom_name, residue_name, chain_id,
                              residue_number, x, y, z, element.upper())
                    self.atoms.append(atom)
                except (ValueError, IndexError) as e:
                    continue
        
        print(f"Parsed {len(self.atoms)} atoms from PDB file")
    
    def load_pdb_file(self, filename: str) -> None:
        """Load PDB file from local filesystem"""
        try:
            with open(filename, 'r') as file:
                pdb_content = file.read()
            self.protein_name = filename.split('/')[-1].split('.')[0]
            self.parse_pdb_content(pdb_content)
        except FileNotFoundError:
            print(f"File {filename} not found")
        except Exception as e:
            print(f"Error loading file: {e}")
    
    def load_pdb_online(self, pdb_id: str) -> None:
        """Load PDB file from online database"""
        self.protein_name = pdb_id.upper()
        pdb_content = self.download_pdb(pdb_id)
        if pdb_content:
            self.parse_pdb_content(pdb_content)
    
    def get_backbone_atoms(self) -> List[Atom]:
        """Get only backbone atoms (N, CA, C, O)"""
        backbone_names = {'N', 'CA', 'C', 'O'}
        return [atom for atom in self.atoms if atom.atom_name in backbone_names]
    
    def get_alpha_carbons(self) -> List[Atom]:
        """Get only alpha carbon atoms"""
        return [atom for atom in self.atoms if atom.atom_name == 'CA']
    
    def visualize_3d(self, style='all_atoms', color_by='element', 
                    chains=None, size_factor=20, alpha=0.8, show_bonds=False):
        """
        Create 3D visualization of protein structure
        
        Parameters:
        - style: 'all_atoms', 'backbone', 'alpha_carbons'
        - color_by: 'element', 'residue', 'chain'
        - chains: list of chain IDs to display (None for all)
        - size_factor: size of atoms in visualization
        - alpha: transparency
        - show_bonds: whether to show bonds between atoms
        """
        if not self.atoms:
            print("No atoms to visualize. Please load a PDB file first.")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Filter atoms based on style
        if style == 'backbone':
            atoms_to_plot = self.get_backbone_atoms()
        elif style == 'alpha_carbons':
            atoms_to_plot = self.get_alpha_carbons()
        else:  # all_atoms
            atoms_to_plot = self.atoms
        
        # Filter by chains if specified
        if chains:
            atoms_to_plot = [atom for atom in atoms_to_plot if atom.chain_id in chains]
        
        if not atoms_to_plot:
            print("No atoms match the specified criteria")
            return
        
        # Extract coordinates and colors
        x = [atom.x for atom in atoms_to_plot]
        y = [atom.y for atom in atoms_to_plot]
        z = [atom.z for atom in atoms_to_plot]
        
        # Determine colors
        colors = []
        for atom in atoms_to_plot:
            if color_by == 'element':
                color = self.element_colors.get(atom.element, self.element_colors['default'])
            elif color_by == 'residue':
                color = self.residue_colors.get(atom.residue_name, self.residue_colors['default'])
            elif color_by == 'chain':
                # Generate colors for different chains
                chain_colors = plt.cm.Set3(np.linspace(0, 1, len(set(atom.chain_id for atom in atoms_to_plot))))
                unique_chains = sorted(list(set(atom.chain_id for atom in atoms_to_plot)))
                chain_idx = unique_chains.index(atom.chain_id) if atom.chain_id in unique_chains else 0
                color = chain_colors[chain_idx % len(chain_colors)]
            else:
                color = '#909090'
            colors.append(color)
        
        # Plot atoms
        scatter = ax.scatter(x, y, z, c=colors, s=size_factor, alpha=alpha, edgecolors='black', linewidth=0.1)
        
        # Add bonds for backbone visualization
        if show_bonds and style in ['backbone', 'alpha_carbons']:
            self._add_bonds(ax, atoms_to_plot)
        
        # Customize plot
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Protein Structure: {self.protein_name}\nStyle: {style}, Colored by: {color_by}')
        
        # Equal aspect ratio
        max_range = max([max(x) - min(x), max(y) - min(y), max(z) - min(z)]) / 2.0
        mid_x, mid_y, mid_z = (max(x) + min(x)) / 2, (max(y) + min(y)) / 2, (max(z) + min(z)) / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()
    
    def _add_bonds(self, ax, atoms_to_plot):
        """Add bonds between consecutive atoms in the backbone"""
        for i in range(len(atoms_to_plot) - 1):
            atom1 = atoms_to_plot[i]
            atom2 = atoms_to_plot[i + 1]
            
            # Only connect atoms from the same chain and consecutive residues
            if (atom1.chain_id == atom2.chain_id and 
                abs(atom1.residue_number - atom2.residue_number) <= 1):
                
                distance = np.sqrt((atom1.x - atom2.x)**2 + 
                                 (atom1.y - atom2.y)**2 + 
                                 (atom1.z - atom2.z)**2)
                
                # Only draw bond if atoms are close enough (reasonable bond length)
                if distance < 4.0:  # 4 Angstroms cutoff
                    ax.plot([atom1.x, atom2.x], [atom1.y, atom2.y], [atom1.z, atom2.z], 
                           'gray', alpha=0.6, linewidth=0.8)
    
    def plot_ramachandran(self):
        """Create Ramachandran plot (phi vs psi angles)"""
        if not self.atoms:
            print("No atoms loaded. Please load a PDB file first.")
            return
        
        # Group atoms by residue
        residues = {}
        for atom in self.atoms:
            key = (atom.chain_id, atom.residue_number)
            if key not in residues:
                residues[key] = {}
            residues[key][atom.atom_name] = atom
        
        phi_angles = []
        psi_angles = []
        
        # Calculate dihedral angles
        for (chain_id, res_num) in sorted(residues.keys()):
            if res_num == 1:  # Skip first residue (no phi angle)
                continue
            
            prev_key = (chain_id, res_num - 1)
            next_key = (chain_id, res_num + 1)
            
            if prev_key not in residues or next_key not in residues:
                continue
            
            current_res = residues[(chain_id, res_num)]
            prev_res = residues[prev_key]
            next_res = residues[next_key]
            
            # Check if all required atoms are present
            required_atoms = ['N', 'CA', 'C']
            if not all(atom in current_res for atom in required_atoms):
                continue
            if 'C' not in prev_res or 'N' not in next_res:
                continue
            
            # Calculate phi angle (C_i-1, N_i, CA_i, C_i)
            phi = self._calculate_dihedral(prev_res['C'], current_res['N'], 
                                         current_res['CA'], current_res['C'])
            
            # Calculate psi angle (N_i, CA_i, C_i, N_i+1)
            psi = self._calculate_dihedral(current_res['N'], current_res['CA'], 
                                         current_res['C'], next_res['N'])
            
            phi_angles.append(np.degrees(phi))
            psi_angles.append(np.degrees(psi))
        
        if not phi_angles:
            print("Could not calculate dihedral angles. Check if backbone atoms are present.")
            return
        
        # Create Ramachandran plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(phi_angles, psi_angles, alpha=0.6, s=30)
        ax.set_xlabel('Phi angle (degrees)')
        ax.set_ylabel('Psi angle (degrees)')
        ax.set_title(f'Ramachandran Plot - {self.protein_name}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        
        # Add reference lines
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Plotted {len(phi_angles)} residues in Ramachandran plot")
    
    def _calculate_dihedral(self, atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom) -> float:
        """Calculate dihedral angle between four atoms"""
        # Convert to numpy arrays
        p1 = np.array([atom1.x, atom1.y, atom1.z])
        p2 = np.array([atom2.x, atom2.y, atom2.z])
        p3 = np.array([atom3.x, atom3.y, atom3.z])
        p4 = np.array([atom4.x, atom4.y, atom4.z])
        
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        
        # Calculate normals
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        
        # Normalize
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        
        # Calculate dihedral angle
        cos_angle = np.dot(n1, n2)
        cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
        
        angle = np.arccos(cos_angle)
        
        # Determine sign
        if np.dot(np.cross(n1, n2), v2) < 0:
            angle = -angle
        
        return angle
    
    def get_info(self) -> Dict:
        """Get information about the loaded protein"""
        if not self.atoms:
            return {"error": "No protein loaded"}
        
        chains = list(set(atom.chain_id for atom in self.atoms))
        residues = list(set(f"{atom.chain_id}:{atom.residue_number}" for atom in self.atoms))
        elements = list(set(atom.element for atom in self.atoms))
        
        # Calculate center of mass
        x_coords = [atom.x for atom in self.atoms]
        y_coords = [atom.y for atom in self.atoms]
        z_coords = [atom.z for atom in self.atoms]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        center_z = sum(z_coords) / len(z_coords)
        
        return {
            "protein_name": self.protein_name,
            "total_atoms": len(self.atoms),
            "chains": sorted(chains),
            "num_residues": len(residues),
            "elements": sorted(elements),
            "center_of_mass": (round(center_x, 2), round(center_y, 2), round(center_z, 2)),
            "x_range": (round(min(x_coords), 2), round(max(x_coords), 2)),
            "y_range": (round(min(y_coords), 2), round(max(y_coords), 2)),
            "z_range": (round(min(z_coords), 2), round(max(z_coords), 2))
        }

# Example usage and demonstration
if __name__ == "__main__":
    # Create visualizer instance
    visualizer = ProteinVisualizer()
    
    print("Protein Structure Visualizer")
    print("=" * 40)
    
    # Example 1: Load a small protein from PDB (Crambin - 1CRN)
    print("Loading crambin protein (1CRN) from PDB database...")
    visualizer.load_pdb_online("1CRN")
    
    # Display protein information
    info = visualizer.get_info()
    print("\nProtein Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example visualizations
    print("\n" + "=" * 40)
    print("Creating visualizations...")
    
    # 1. All atoms colored by element
    print("1. All atoms visualization (colored by element)")
    visualizer.visualize_3d(style='all_atoms', color_by='element', size_factor=30)
    
    # 2. Backbone atoms only
    print("2. Backbone visualization with bonds")
    visualizer.visualize_3d(style='backbone', color_by='residue', 
                           size_factor=40, show_bonds=True)
    
    # 3. Alpha carbons only (simplified view)
    print("3. Alpha carbons only (simplified view)")
    visualizer.visualize_3d(style='alpha_carbons', color_by='chain', 
                           size_factor=60, show_bonds=True)
    
    # 4. Ramachandran plot
    print("4. Ramachandran plot")
    visualizer.plot_ramachandran()
    
    print("\nVisualization complete!")
    print("\nTo use with your own PDB files:")
    print("  visualizer.load_pdb_file('path/to/your/file.pdb')")
    print("  visualizer.visualize_3d()")
    
    print("\nTo load other proteins from PDB database:")
    print("  visualizer.load_pdb_online('PDB_ID')  # e.g., '1A3N', '2HHB'")