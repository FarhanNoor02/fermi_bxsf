import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
import os
from itertools import combinations

def read_bxsf(filename):
    data = {
        'fermi_energy': None,
        'band_data': [],
        'grid_dimensions': None,
        'origin': None,
        'vectors': None
    }

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        if 'Fermi Energy:' in line:
            data['fermi_energy'] = float(line.split(':')[1].strip())
            i += 1
        elif line == 'BEGIN_BLOCK_BANDGRID_3D':
            i += 1
            while i < len(lines) and lines[i] != 'BANDGRID_3D_BANDS':
                i += 1
            i += 1

            n_bands = int(lines[i])
            i += 1
            data['grid_dimensions'] = list(map(int, lines[i].split()))
            i += 1
            data['origin'] = list(map(float, lines[i].split()))
            i += 1

            data['vectors'] = []
            for _ in range(3):
                data['vectors'].append(list(map(float, lines[i].split())))
                i += 1

            for band_idx in range(n_bands):
                while i < len(lines) and not lines[i].startswith('BAND:'):
                    i += 1
                if i >= len(lines): break
                i += 1

                band_data = []
                total_points = np.prod(data['grid_dimensions'])
                while len(band_data) < total_points and i < len(lines):
                    if lines[i].startswith('BAND:'): break
                    band_data.extend(map(float, lines[i].split()))
                    i += 1

                if len(band_data) == total_points:
                    band_data = np.array(band_data).reshape(*data['grid_dimensions'])
                    data['band_data'].append(band_data)
        else:
            i += 1

    return data

def get_brillouin_zone_edges(reciprocal_lattice):
    """Calculate Brillouin zone edges using Wigner-Seitz method"""
    # Generate nearest neighbor reciprocal lattice points
    offsets = np.array(list(combinations([-1,0,1], 3)))
    neighbor_points = offsets @ reciprocal_lattice
    
    # Find perpendicular bisector planes
    planes = []
    for point in neighbor_points:
        if np.allclose(point, [0,0,0]):
            continue
        normal = point / np.linalg.norm(point)
        distance = np.dot(normal, point) / 2
        planes.append((normal, distance))
    
    return planes

def plot_brillouin_zone(ax, reciprocal_lattice, lattice_type='cubic'):
    """Plot Brillouin zone edges based on lattice type"""
    if lattice_type == 'cubic':
        # Simple cubic BZ is a cube
        vertices = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ]) @ reciprocal_lattice
        
        edges = [
            (0,1), (1,2), (2,3), (3,0),  # Bottom face
            (4,5), (5,6), (6,7), (7,4),  # Top face
            (0,4), (1,5), (2,6), (3,7)   # Vertical edges
        ]
        
    elif lattice_type == 'hexagonal':
        # Hexagonal BZ is more complex
        a = np.linalg.norm(reciprocal_lattice[0])
        c = np.linalg.norm(reciprocal_lattice[2])
        
        vertices = np.array([
            [0, -2/(3*a), -0.5/c],  # L
            [1/(np.sqrt(3)*a), -1/(3*a), -0.5/c],  # L
            [1/(np.sqrt(3)*a), 1/(3*a), -0.5/c],   # L
            [0, 2/(3*a), -0.5/c],    # L
            [-1/(np.sqrt(3)*a), 1/(3*a), -0.5/c],  # L
            [-1/(np.sqrt(3)*a), -1/(3*a), -0.5/c], # L
            
            [0, -2/(3*a), 0.5/c],    # U
            [1/(np.sqrt(3)*a), -1/(3*a), 0.5/c],   # U
            [1/(np.sqrt(3)*a), 1/(3*a), 0.5/c],    # U
            [0, 2/(3*a), 0.5/c],     # U
            [-1/(np.sqrt(3)*a), 1/(3*a), 0.5/c],   # U
            [-1/(np.sqrt(3)*a), -1/(3*a), 0.5/c],  # U
            
            [0, 0, -0.5/c],  # Bottom center
            [0, 0, 0.5/c]     # Top center
        ])
        
        edges = [
            (0,1), (1,2), (2,3), (3,4), (4,5), (5,0),  # Bottom hexagon
            (6,7), (7,8), (8,9), (9,10), (10,11), (11,6),  # Top hexagon
            (0,6), (1,7), (2,8), (3,9), (4,10), (5,11),  # Vertical edges
            (12,0), (12,1), (12,2), (12,3), (12,4), (12,5),  # Bottom center
            (13,6), (13,7), (13,8), (13,9), (13,10), (13,11)  # Top center
        ]
        
    else:
        # Default to simple cubic if lattice type not recognized
        return plot_brillouin_zone(ax, reciprocal_lattice, 'cubic')
    
    # Plot edges
    for edge in edges:
        ax.plot3D(*zip(vertices[edge[0]], vertices[edge[1]]), 
                 color='black', lw=1.2, alpha=0.7)
    
    return vertices

def get_high_symmetry_points(lattice_type='cubic'):
    """Return standard high symmetry points for different lattices"""
    if lattice_type == 'cubic':
        return {
            r'$\Gamma$': [0.0, 0.0, 0.0],
            'X': [0.5, 0.0, 0.0],
            'R': [0.5, 0.5, 0.5],
            'M': [0.5, 0.5, 0.0]
        }
    elif lattice_type == 'hexagonal':
        return {
            r'$\Gamma$': [0.0, 0.0, 0.0],
            'M': [0.5, 0.0, 0.0],
            'K': [1/3, 1/3, 0.0],
            'A': [0.0, 0.0, 0.5],
            'L': [0.5, 0.0, 0.5],
            'H': [1/3, 1/3, 0.5]
        }
    elif lattice_type == 'tetragonal':
        return {
            r'$\Gamma$': [0.0, 0.0, 0.0],
            'X': [0.5, 0.0, 0.0],
            'M': [0.5, 0.5, 0.0],
            'Z': [0.0, 0.0, 0.5],
            'R': [0.5, 0.0, 0.5],
            'A': [0.5, 0.5, 0.5]
        }
    else:  # default to cubic
        return get_high_symmetry_points('cubic')

def plot_fermi_surface(bxsf_data, band_index=0, isovalue=None, 
                      show_plot=True, lattice_type='cubic'):
    if isovalue is None:
        isovalue = bxsf_data['fermi_energy']

    if band_index >= len(bxsf_data['band_data']):
        print(f"Error: Band index {band_index} is out of range.")
        return

    band_data = bxsf_data['band_data'][band_index]
    nx, ny, nz = bxsf_data['grid_dimensions']
    vecs = np.array(bxsf_data['vectors'])
    recip_lattice = 2 * np.pi * np.linalg.inv(vecs).T  # Reciprocal vectors

    verts, faces, _, _ = marching_cubes(band_data, isovalue)

    # Normalize and convert to reciprocal space
    verts[:, 0] /= (nx - 1)
    verts[:, 1] /= (ny - 1)
    verts[:, 2] /= (nz - 1)

    # Shift coordinates so Gamma point is at center (0.5, 0.5, 0.5) -> (0, 0, 0)
    verts = verts - 0.5

    verts_recip = verts @ recip_lattice

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Fermi surface
    ax.plot_trisurf(verts_recip[:, 0], verts_recip[:, 1], faces, verts_recip[:, 2],
                    cmap='viridis', edgecolor='none', alpha=0.8)

    # Plot Brillouin zone
    bz_vertices = plot_brillouin_zone(ax, recip_lattice, lattice_type)

    # Plot high symmetry points
    high_sym_points = get_high_symmetry_points(lattice_type)
    for label, coord in high_sym_points.items():
        k_cart = np.dot(coord, recip_lattice)
        ax.scatter(*k_cart, color='red', s=50)
        ax.text(*k_cart, label, fontsize=20, weight='bold', color='black')

    # Set viewing angle and limits
    ax.set_xlim([bz_vertices[:,0].min(), bz_vertices[:,0].max()])
    ax.set_ylim([bz_vertices[:,1].min(), bz_vertices[:,1].max()])
    ax.set_zlim([bz_vertices[:,2].min(), bz_vertices[:,2].max()])
    
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()
    plt.tight_layout()
    
    if show_plot:
        plt.show()

def main():
    filename = input("Enter path to BXSF file: ").strip('"').strip("'")
    band_index = int(input("Enter band index (default 0): ") or "0")
    isovalue_input = input("Enter isovalue (default Fermi energy): ")
    isovalue = float(isovalue_input) if isovalue_input else None
    lattice_type = input("Enter lattice type (cubic, hexagonal, tetragonal, default cubic): ") or "cubic"

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    print(f"Reading {filename}...")
    bxsf_data = read_bxsf(filename)
    print(f"Successfully read file with {len(bxsf_data['band_data'])} bands")
    print(f"Fermi energy: {bxsf_data['fermi_energy']} eV")
    print(f"Grid dimensions: {bxsf_data['grid_dimensions']}")

    plot_fermi_surface(bxsf_data, band_index=band_index, 
                      isovalue=isovalue, lattice_type=lattice_type.lower())

if __name__ == "__main__":
    main()
