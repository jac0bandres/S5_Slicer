import numpy as np
import tetgen
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
import pickle
import time
import base64
import os
import pyvista as pv
import subprocess
import configparser
import json

import warnings
warnings.simplefilter('ignore')
CURA_LOGS_DIR = os.path.abspath('./cura_logs')

def update_config():
    cfgp = configparser.ConfigParser()

    cfgp.read('config/easys4.ini')
    with open('config/core.def.json', 'r') as jfh:
        slicer_config = json.load(jfh)

    curr_layer_height = slicer_config['settings']['global']['all']['layer_height']
    layer_height = cfgp['SLICER']['LayerHeight']
    if curr_layer_height != layer_height:
        slicer_config['settings']['global']['all']['layer_height'] = layer_height

    with open('config/core.def.json', 'w') as jfh:
        json.dump(slicer_config, jfh, indent=4)

def get_config():
    cfgp = configparser.ConfigParser()
    cfgp.read('config/easys4.ini')
    cura_path = cfgp['PATHS']['curaengine']
    return cura_path

def update_cura_config(src):
    with open(src, 'r') as jfh:
        slicer_config = json.load(jfh)

    cfgp = configparser.ConfigParser()
    cfgp.read('config/easys4.ini')

    for setting_name, data in slicer_config['settings']['global']['all'].items():
        if isinstance(data, dict):
            val = data.get('value') or data.get('default_value')
        else:
            val = data
        
        cfgp['SLICER'][str(setting_name)] = str(val)
    
    with open('config/easys4.ini', 'w') as cfgh:
        cfgp.write(cfgh)

def update_layer_height(layer_height):
    cfgp = configparser.ConfigParser()
    cfgp.read('config/easys4.ini')

    cfgp['SLICER']['layer_height'] = layer_height
    
    with open('config/easys4.ini', 'w') as cfgh:
        cfgp.write(cfgh)

def get_slicer_settings(command):
    cfgp = configparser.ConfigParser()
    cfgp.read('config/easys4.ini')
    for setting_name in cfgp['SLICER']:
        value = cfgp['SLICER'][setting_name]
        command.extend(["-s", f'{setting_name}={value}'])
    
    return command

total_time = time.time()
plotter = pv.Plotter()

def slice(model_path):
    total_t = time.time()
    prev_time = total_t
    if model_path:
        model_name = os.path.basename(model_path).split('.')[0]
        print(f'Slicing {model_name}...')
        model_name = f'{model_name}_{time.time()}'
    else:
        print("No model path specified.")
        exit()

    cura_path = get_config()

    def encode_object(obj):
        return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

    def decode_object(encoded_str):
        return pickle.loads(base64.b64decode(encoded_str))

    up_vector = np.array([0, 0, 1])

    mesh = o3d.io.read_triangle_mesh(model_path)
    input_tet = tetgen.TetGen(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    input_tet.tetrahedralize()

    input_tet = input_tet.grid
    # input_tet = input_tet.scale(1)

    PART_OFFSET = np.array([0., 0., 0.])
    x_min, x_max, y_min, y_max, z_min, z_max = input_tet.bounds
    input_tet.points -= np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]) + PART_OFFSET
    plotter.add_mesh(input_tet, show_edges=True)

    print("Finding cell neighbors") 
    cells_only = input_tet.cells.reshape(-1, 5)[:, 1:]
    num_cells = input_tet.n_cells
    num_points = input_tet.n_points

    cell_indices = np.repeat(np.arange(num_cells), 4)
    point_indices = cells_only.ravel()
    data = np.ones(len(cell_indices), dtype=int)

    C_P = sparse.csr_matrix((data, (cell_indices, point_indices)), shape=(num_cells, num_points))
    A = (C_P @ C_P.T).tocoo()

    mask = A.row != A.col
    rows = A.row[mask]
    cols = A.col[mask]
    counts = A.data[mask]

    sym_mask = rows < cols

    for n_type, condition in [
        ('point', counts >= 1),
        ('edge', counts >= 2),
        ('face', counts == 3)
    ]:
        final_mask = condition & sym_mask
        input_tet.field_data[f'cell_{n_type}_neighbours'] = np.column_stack((
                             cols[final_mask],
                             rows[final_mask]))
        
    # OPT: Build padded_edge_neighbors directly from COO arrays — eliminates two O(n_edges)
    # Python loops (the dict build + the padded array fill).  Strategy: sort all (src,dst)
    # edge pairs by src, then compute each entry's column offset via cumsum, then scatter-fill.
    _ec      = counts >= 2
    _e_src   = rows[_ec]
    _e_dst   = cols[_ec]
    # Both directions + self-loop per cell
    _src_f   = np.concatenate([_e_src, _e_dst, np.arange(num_cells, dtype=np.int32)])
    _dst_f   = np.concatenate([_e_dst, _e_src, np.arange(num_cells, dtype=np.int32)])
    _ord     = np.argsort(_src_f, kind='stable')
    _src_s   = _src_f[_ord];  _dst_s = _dst_f[_ord]
    _nb_cnt  = np.bincount(_src_s, minlength=num_cells)
    _max_nb  = int(_nb_cnt.max())
    padded_edge_neighbors = np.full((num_cells, _max_nb), -1, dtype=np.int32)
    _iptr    = np.concatenate([[0], np.cumsum(_nb_cnt)])
    _col_idx = np.arange(len(_dst_s), dtype=np.int32) - _iptr[_src_s]
    padded_edge_neighbors[_src_s, _col_idx] = _dst_s

    print(f"Neighbors done: {round(time.time()- prev_time, 2)}s")
    prev_time = time.time()

    cell_centers = input_tet.cell_centers().points
    edges = input_tet.field_data["cell_point_neighbours"]

    starts = cell_centers[edges[:, 0]]
    ends = cell_centers[edges[:, 1]]
    distances = np.linalg.norm(starts - ends, axis=1)

    # Opt: SciPy Dijkstra sparse matrix setup
    graph_csr = sparse.coo_matrix((distances, (edges[:, 0], edges[:, 1])), shape=(num_cells, num_cells)).tocsr()
    graph_csr = graph_csr.maximum(graph_csr.T) 

    def update_tet_attributes(tet):
        print("updating tet attributes...")
        curr_t = time.time()
        surface_mesh = tet.extract_surface()
        cell_to_face = decode_object(tet.field_data["cell_to_face"])

        cells = tet.cells.reshape(-1, 5)[:, 1:]
        tet.add_field_data(cells, "cells")
        cell_vertices = tet.points
        tet.add_field_data(cell_vertices, "cell_vertices")
        faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]
        tet.add_field_data(faces, "faces")
        face_vertices = surface_mesh.points
        tet.add_field_data(face_vertices, "face_vertices")

        tet.cell_data['face_normal'] = np.full((tet.number_of_cells, 3), np.nan)
        keys = list(cell_to_face.keys())
        values = list(cell_to_face.values())
        all_face_indices = np.concatenate(values)
        cell_counts = [len(f) for f in values]
        all_cell_ids = np.repeat(keys, cell_counts)
        all_normals = surface_mesh.face_normals[all_face_indices]
        sort_idx = np.lexsort((all_normals[:, 2], all_cell_ids))
        sorted_ids = all_cell_ids[sort_idx]
        sorted_normals = all_normals[sort_idx]
        unique_ids, first_indices = np.unique(sorted_ids, return_index=True)
        tet.cell_data['face_normal'][unique_ids] = sorted_normals[first_indices]
        final_normals = tet.cell_data['face_normal']
        tet.cell_data['face_normal'] = final_normals / np.clip(np.linalg.norm(final_normals, axis=1)[:, None], 1e-15, None)

        tet.cell_data['face_center'] = np.empty((tet.number_of_cells, 3))
        tet.cell_data['face_center'][:,:] = np.nan
        surface_mesh_cell_centers = surface_mesh.cell_centers().points

        all_centers = surface_mesh_cell_centers[all_face_indices]
        sort_idx_centers = np.lexsort((all_centers[:, 2], all_cell_ids))
        sorted_ids_centers = all_cell_ids[sort_idx_centers]
        sorted_centers = all_centers[sort_idx_centers]
        _, first_indices_centers = np.unique(sorted_ids_centers, return_index=True)

        tet.cell_data['face_center'] = np.full((tet.number_of_cells, 3), np.nan)
        tet.cell_data['face_center'][keys] = sorted_centers[first_indices_centers]

        tet.cell_data['cell_center'] = tet.cell_centers().points
        bottom_cell_threshold = np.nanmin(tet.cell_data['face_center'][:, 2]) + 0.3
        bottom_cells_mask = tet.cell_data['face_center'][:, 2] < bottom_cell_threshold
        tet.cell_data['is_bottom'] = bottom_cells_mask
        bottom_cells = np.where(bottom_cells_mask)[0]
        face_normals = tet.cell_data['face_normal'].copy()
        face_normals[bottom_cells_mask] = np.nan
        dots = np.sum(face_normals * up_vector, axis=1)
        tet.cell_data['overhang_angle'] = np.arccos(np.clip(dots, -1.0, 1.0))

        print(f'{time.time()-curr_t}')
        return tet

    def calculate_tet_attributes(tet):
        print("calc tet attributes...")
        curr_t = time.time()
        surface_mesh = tet.extract_surface()
        cells = tet.cells.reshape(-1, 5)[:, 1:]
        tet.add_field_data(cells, "cells")
        tet.add_field_data(tet.points, "cell_vertices")
        
        faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]
        tet.add_field_data(faces, "faces")
        tet.add_field_data(surface_mesh.points, "face_vertices")

        from collections import defaultdict
        cell_to_face = defaultdict(list)
        face_to_cell = defaultdict(list)
        
        orig_cell_ids = surface_mesh.cell_data['vtkOriginalCellIds']
        for face_idx, tet_cell_idx in enumerate(orig_cell_ids):
            cell_to_face[tet_cell_idx].append(face_idx)
            face_to_cell[face_idx].append(tet_cell_idx)

        tet.add_field_data(encode_object(dict(cell_to_face)), "cell_to_face")
        tet.add_field_data(encode_object(dict(face_to_cell)), "face_to_cell")

        tet.cell_data['has_face'] = np.zeros(tet.number_of_cells, dtype=int)
        tet.cell_data['has_face'][list(cell_to_face.keys())] = 1

        tet = update_tet_attributes(tet)
        
        bottom_cells_mask = tet.cell_data['is_bottom']
        bottom_cells = np.where(bottom_cells_mask)[0]
        tet.cell_data['overhang_angle'][bottom_cells] = np.nan

        print(time.time()-curr_t)

        return tet, bottom_cells_mask, bottom_cells

    input_tet, _, bottom_cells = calculate_tet_attributes(input_tet)
    undeformed_tet = input_tet.copy()

    def calculate_path_length_to_base_gradient(tet, MAX_OVERHANG, INITIAL_ROTATION_FIELD_SMOOTHING, SET_INITIAL_ROTATION_TO_ZERO):
        print('calculate_path_length_to_base_gradient')
        curr_t = time.time()
        path_length_to_base_gradient = np.zeros((tet.number_of_cells))
        n_cells_local = tet.number_of_cells

        distances_to_bottom, predecessors, _ = dijkstra(
            csgraph=graph_csr, directed=False, indices=list(bottom_cells),
            return_predecessors=True, min_only=True
        )
        cell_distance_to_bottom = distances_to_bottom

        # OPT: Boolean mask for O(1) membership tests — replaces O(n) `in bottom_cells` checks.
        is_bottom_mask = np.zeros(n_cells_local, dtype=bool)
        is_bottom_mask[bottom_cells] = True

        # OPT: Vectorized predecessor tracing — follow all cells toward their bottom ancestor
        # simultaneously using NumPy operations instead of per-cell Python while-loops.
        curr_nodes = np.arange(n_cells_local, dtype=np.int32)
        for _ in range(n_cells_local):          # worst-case depth of Dijkstra tree
            prev_nodes = predecessors[curr_nodes]
            # Move toward root only if: predecessor exists AND current node is not already bottom
            needs_move = (prev_nodes >= 0) & (~is_bottom_mask[curr_nodes])
            if not needs_move.any():
                break
            curr_nodes[needs_move] = prev_nodes[needs_move].astype(np.int32)

        closest_bottom_cell_indices = curr_nodes.copy()
        # Cells that never reached a bottom node fall back to bottom_cells[0]
        if len(bottom_cells) > 0:
            not_at_bottom = ~is_bottom_mask[curr_nodes]
            closest_bottom_cell_indices[not_at_bottom] = bottom_cells[0]

        # OPT: Fully vectorized gradient computation using pre-built padded_edge_neighbors.
        # Avoids O(n) Python loop with per-cell SVD.
        finite_mask = ~np.isinf(cell_distance_to_bottom)
        finite_cells = np.where(finite_mask)[0]

        if len(finite_cells) == 0:
            if not SET_INITIAL_ROTATION_TO_ZERO:
                path_length_to_base_gradient[path_length_to_base_gradient == 0] = np.nan
            tet.cell_data["path_length_to_base_gradient"] = path_length_to_base_gradient
            print(time.time()-curr_t)
            return path_length_to_base_gradient

        cc = tet.cell_data["cell_center"]

        # Get padded neighbors for finite cells and their distances
        pn = padded_edge_neighbors[finite_cells]                          # (n_f, max_nb)
        valid_nb = pn >= 0                                                 # (n_f, max_nb)
        clipped_pn = np.clip(pn, 0, n_cells_local - 1)
        dist_nb = np.where(valid_nb, cell_distance_to_bottom[clipped_pn], np.inf)  # (n_f, max_nb)
        finite_nb = ~np.isinf(dist_nb)                                    # valid AND finite-distance
        valid_count = finite_nb.sum(axis=1)                               # (n_f,)

        # Split cells by gradient method
        use_dir = valid_count < 3                                          # direction-to-bottom
        use_svd = ~use_dir                                                 # SVD plane fit

        # --- Branch A: direction-to-bottom (< 3 valid finite neighbors) ---
        if use_dir.any():
            dir_idx = finite_cells[use_dir]
            bottom_locs  = cc[closest_bottom_cell_indices[dir_idx], :2]   # (n_d, 2)
            cell_locs    = cc[dir_idx, :2]                                 # (n_d, 2)
            d2b          = bottom_locs - cell_locs                         # (n_d, 2)
            n_d2b        = np.linalg.norm(d2b, axis=1, keepdims=True)
            d2b_norm     = np.where(n_d2b > 0, d2b / n_d2b, d2b)

            c_rad        = cell_locs.copy()
            c_rad_n      = np.linalg.norm(c_rad, axis=1, keepdims=True)
            c_rad_norm   = np.where(c_rad_n > 0, c_rad / c_rad_n, c_rad)

            dots         = np.sum(c_rad_norm * d2b_norm, axis=1)
            opt_rot      = dots / (np.abs(dots) + 1e-8)
            opt_rot      = np.where(np.isnan(opt_rot), 0.0, opt_rot)
            path_length_to_base_gradient[dir_idx] = opt_rot

        # --- Branch B: SVD plane fit (>= 3 valid finite neighbors) ---
        if use_svd.any():
            svd_pn    = pn[use_svd]                                        # (n_s, max_nb)
            svd_valid = finite_nb[use_svd]                                 # (n_s, max_nb)
            svd_dist  = dist_nb[use_svd]                                   # (n_s, max_nb)
            svd_cells = finite_cells[use_svd]

            # Gather local cell centers (x, y) for each neighbor slot
            clipped_svd_pn = np.clip(svd_pn, 0, n_cells_local - 1)
            local_cc = cc[clipped_svd_pn, :2]                             # (n_s, max_nb, 2)

            # Build local point matrix [x, y, path_length] — zero-out invalid slots
            pts = np.concatenate(
                [local_cc, svd_dist[:, :, None]], axis=2
            )                                                              # (n_s, max_nb, 3)
            mask3 = svd_valid[:, :, None]
            pts   = np.where(mask3, pts, 0.0)

            # Masked mean and centred points
            cnt   = svd_valid.sum(axis=1, keepdims=True).clip(min=1)      # (n_s, 1)
            ctr   = pts.sum(axis=1) / cnt                                 # (n_s, 3)
            x_c   = pts - ctr[:, None, :]                                 # (n_s, max_nb, 3)
            x_c   = np.where(mask3, x_c, 0.0)

            # Batch covariance and SVD — M = x.T @ x per cell, shape (n_s, 3, 3)
            M     = np.einsum('nki,nkj->nij', x_c, x_c)
            U, _, _ = np.linalg.svd(M)
            plane_normals = U[:, :, -1]                                   # last col = min variance direction

            cc_2d = cc[svd_cells, :2]
            cc_n  = np.linalg.norm(cc_2d, axis=1, keepdims=True) + 1e-8
            cc_norm = cc_2d / cc_n
            grad  = np.sum(cc_norm * plane_normals[:, :2], axis=1)
            grad  = np.where(np.isnan(grad), 0.0, grad)
            path_length_to_base_gradient[svd_cells] = grad

        if not SET_INITIAL_ROTATION_TO_ZERO:
            path_length_to_base_gradient[path_length_to_base_gradient == 0] = np.nan
        tet.cell_data["path_length_to_base_gradient"] = path_length_to_base_gradient
        print(time.time()-curr_t)
        return path_length_to_base_gradient


    def calculate_initial_rotation_field(tet, MAX_OVERHANG, ROTATION_MULTIPLIER, STEEP_OVERHANG_COMPENSATION, INITIAL_ROTATION_FIELD_SMOOTHING, SET_INITIAL_ROTATION_TO_ZERO, MAX_POS_ROTATION, MAX_NEG_ROTATION):
        print('calculate_intial_rotation_field')
        curr_t = time.time()
        initial_rotation_field = np.abs(np.deg2rad(90+MAX_OVERHANG) - tet.cell_data['overhang_angle'])
        path_length_to_base_gradient = calculate_path_length_to_base_gradient(tet, MAX_OVERHANG, INITIAL_ROTATION_FIELD_SMOOTHING, SET_INITIAL_ROTATION_TO_ZERO)

        if STEEP_OVERHANG_COMPENSATION:
            in_air_mask = tet.cell_data.get("in_air", np.zeros(tet.number_of_cells, dtype=bool))
            initial_rotation_field[in_air_mask] += 2 * (np.deg2rad(180) - tet.cell_data['overhang_angle'][in_air_mask])

        initial_rotation_field *= path_length_to_base_gradient
        initial_rotation_field = np.clip(initial_rotation_field*ROTATION_MULTIPLIER, -np.deg2rad(360), np.deg2rad(360))
        initial_rotation_field = np.clip(initial_rotation_field, MAX_NEG_ROTATION, MAX_POS_ROTATION)

        tet.cell_data["initial_rotation_field"] = initial_rotation_field
        print(time.time() - curr_t)

        return initial_rotation_field

    def calculate_rotation_matrices(tet, rotation_field):
        print('calculate_rotation_matrices')
        curr_t = time.time()
        tangential_vectors = np.cross( np.array([0, 0, 1]), tet.cell_data["cell_center"][:, :2])
        tangential_vectors /= np.linalg.norm(tangential_vectors, axis=1)[:, None]
        tangential_vectors[np.isnan(tangential_vectors).any(axis=1)] = [1, 0, 0]

        rotation_matrices = R.from_rotvec(rotation_field[:, None] * tangential_vectors).as_matrix()
        print(time.time() - curr_t)

        return rotation_matrices

    def optimize_rotations(tet, NEIGHBOUR_LOSS_WEIGHT, MAX_OVERHANG, ROTATION_MULTIPLIER, ITERATIONS, SAVE_GIF, STEEP_OVERHANG_COMPENSATION, INITIAL_ROTATION_FIELD_SMOOTHING, SET_INITIAL_ROTATION_TO_ZERO, MAX_POS_ROTATION, MAX_NEG_ROTATION):
        print('optimized_rotations')
        curr_t = time.time()
        initial_rotation_field = calculate_initial_rotation_field(tet, MAX_OVERHANG, ROTATION_MULTIPLIER, STEEP_OVERHANG_COMPENSATION, INITIAL_ROTATION_FIELD_SMOOTHING, SET_INITIAL_ROTATION_TO_ZERO, MAX_POS_ROTATION, MAX_NEG_ROTATION)

        # OPT: Replace TRF iterative least_squares with a single direct sparse solve.
        #
        # Why this works: the original objective is Laplacian smoothing —
        #   W * Σ_(i,j face-neighbors) (r_i − r_j)²  +  Σ_(valid i) (r_i − init_i)²
        # This is purely quadratic, so its minimiser satisfies the linear system:
        #   (W · L_graph  +  diag(valid_mask)) r  =  diag(valid_mask) · r_init
        # where L_graph is the weighted graph Laplacian over face-neighbor pairs.
        #
        # Previously least_squares rebuilt a full COO→CSR Jacobian on each of up to
        # ITERATIONS TRF evaluations.  spsolve does one UMFPACK LU factorisation and
        # back-substitution — typically < 0.1 s vs 20 + s.

        from scipy.sparse.linalg import spsolve as _spsolve

        cell_face_nb = tet.field_data["cell_face_neighbours"]
        n_c          = tet.number_of_cells
        valid_mask   = ~np.isnan(initial_rotation_field)
        W            = float(NEIGHBOUR_LOSS_WEIGHT)

        ea = cell_face_nb[:, 0]          # one endpoint of each unique face-edge
        eb = cell_face_nb[:, 1]
        n_e = len(ea)

        # Degree = number of face-neighbours per cell (each undirected edge counted once,
        # so we add 1 to both endpoints)
        degree = np.zeros(n_c, dtype=np.float64)
        np.add.at(degree, ea, 1.0)
        np.add.at(degree, eb, 1.0)

        # Diagonal: W * degree + anchor-weight(1 if valid else 0) + tiny eps for stability
        diag_v = W * degree + valid_mask.astype(np.float64) + 1e-10

        all_r = np.concatenate([ea, eb, np.arange(n_c, dtype=np.int32)])
        all_c = np.concatenate([eb, ea, np.arange(n_c, dtype=np.int32)])
        all_d = np.concatenate([-W * np.ones(n_e), -W * np.ones(n_e), diag_v])

        A_rot = sparse.coo_matrix((all_d, (all_r, all_c)),
                                   shape=(n_c, n_c), dtype=np.float64).tocsr()
        b_rot = np.where(valid_mask, np.nan_to_num(initial_rotation_field, nan=0.0), 0.0)

        rotation_field = _spsolve(A_rot, b_rot)
        print(time.time() - curr_t)
        return rotation_field

    NEIGHBOUR_LOSS_WEIGHT = 20 
    MAX_OVERHANG = 30          
    ROTATION_MULTIPLIER = 2   
    SET_INITIAL_ROTATION_TO_ZERO = False 
    INITIAL_ROTATION_FIELD_SMOOTHING = 30
    MAX_POS_ROTATION = np.deg2rad(360) 
    MAX_NEG_ROTATION = np.deg2rad(-360) 
    ITERATIONS = 100
    SAVE_GIF = True
    STEEP_OVERHANG_COMPENSATION = True

    rotation_field = optimize_rotations(
        undeformed_tet,
        NEIGHBOUR_LOSS_WEIGHT,
        MAX_OVERHANG,
        ROTATION_MULTIPLIER,
        ITERATIONS,
        SAVE_GIF,
        STEEP_OVERHANG_COMPENSATION,
        INITIAL_ROTATION_FIELD_SMOOTHING,
        SET_INITIAL_ROTATION_TO_ZERO,
        MAX_POS_ROTATION,
        MAX_NEG_ROTATION
    )

    N = np.eye(4) - 1/4 * np.ones((4, 4)) 

    def calculate_deformation(tet, rotation_field, ITERATIONS, SAVE_GIF):
        """Deformation via parallel sparse lsqr — replaces iterative TRF.

        Original issue: least_squares (TRF) rebuilt a COO→CSR Jacobian on every
        one of up to 1000 function evaluations.  It also had a formulation bug —
        returning per-cell ||M||_F² as residuals made TRF minimise the quartic
        Σ(||M||²)² instead of the correct quadratic Σ||M||².

        The correct objective is a sparse LINEAR least-squares problem that
        decomposes into 3 INDEPENDENT systems (one per spatial dimension):

            min_vj  ||A_deform @ v_j  −  b_j||²     j ∈ {x, y, z}

        A_deform is built ONCE from the mesh topology (never rebuilt per-iter).
        lsqr with a warm-start from the current vertex positions typically
        converges in < 30 iterations vs 1000 TRF calls, with no Jacobian
        allocation in the loop.  The 3 independent solves run in parallel threads
        (scipy CSR matvec releases the GIL), giving another ~3× speedup.

        Upgrade paths for very large models:
          • scikit-sparse CHOLMOD: A^T A direct factorisation — eliminates
            iterations entirely, beats lsqr once n_pts > ~50k.
          • PyPardiso (Intel MKL PARDISO): fastest CPU direct solver.
          • CuPy cuSPARSE: GPU sparse ops, ~10–100× on large models.
        """
        print('calculate_deformations')
        curr_t = time.time()

        from concurrent.futures import ThreadPoolExecutor

        n_cells = tet.number_of_cells
        n_pts   = tet.number_of_points
        cells   = tet.field_data["cells"]           # (n_cells, 4)

        rotation_matrices = calculate_rotation_matrices(tet, rotation_field)

        # Pre-compute per-cell rotation targets
        # old_verts_transformed[c, j, i] = target for cell c, dim j, local vertex i
        old_verts = tet.field_data["cell_vertices"][cells]   # (n_cells, 4, 3)
        old_verts_transformed = np.einsum(
            'ijk,ikl->ijl',
            rotation_matrices,
            (N @ old_verts).transpose(0, 2, 1)
        )  # (n_cells, 3, 4)

        # Build sparse A once — depends only on topology + N, never changes per iter.
        # A[c*4 + i,  cells[c, k]]  =  N[i, k]
        # N = I4 − 0.25·ones4  →  N[i,k] = 0.75 if i==k else −0.25
        c_idx = np.arange(n_cells)
        _rows, _cols, _vals = [], [], []
        for i in range(4):
            for k in range(4):
                _rows.append(c_idx * 4 + i)
                _cols.append(cells[:, k])
                _vals.append(np.full(n_cells, 0.75 if i == k else -0.25,
                                     dtype=np.float64))
        A_deform = sparse.coo_matrix(
            (np.concatenate(_vals),
             (np.concatenate(_rows), np.concatenate(_cols))),
            shape=(n_cells * 4, n_pts), dtype=np.float64
        ).tocsr()

        # RHS: B[c*4 + i, j] = old_verts_transformed[c, j, i]
        B_target = np.ascontiguousarray(
            old_verts_transformed.transpose(0, 2, 1).reshape(n_cells * 4, 3)
        )

        # Parallel lsqr: 3 independent dims solved in parallel threads.
        # Warm-start from current positions → typically converges in < 30 iters.
        # scipy CSR matvec releases the GIL, enabling real thread parallelism.
        x0 = tet.points   # (n_pts, 3) — current vertex positions

        def _lsqr_dim(j):
            sol = sparse.linalg.lsqr(
                A_deform,
                B_target[:, j],
                x0=x0[:, j],
                atol=1e-7, btol=1e-7,
                iter_lim=ITERATIONS,
                show=False
            )
            print(f"  dim {j} ({('x','y','z')[j]}): {sol[2]} iters, "
                  f"residual {sol[3]:.2e}")
            return sol[0]   # (n_pts,)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = [pool.submit(_lsqr_dim, j) for j in range(3)]
            new_verts = np.stack([f.result() for f in futs], axis=1)  # (n_pts, 3)

        print(time.time() - curr_t)
        return new_verts

    ITERATIONS = 1000
    SAVE_GIF = True
    new_vertices = calculate_deformation(undeformed_tet, rotation_field, ITERATIONS, SAVE_GIF)
    deformed_tet = pv.UnstructuredGrid(undeformed_tet.cells, np.full(undeformed_tet.number_of_cells, pv.CellType.TETRA), new_vertices)

    for key in undeformed_tet.field_data.keys():
        deformed_tet.field_data[key] = undeformed_tet.field_data[key]
    for key in undeformed_tet.cell_data.keys():
        deformed_tet.cell_data[key] = undeformed_tet.cell_data[key]
    deformed_tet = update_tet_attributes(deformed_tet)

    x_min, x_max, y_min, y_max, z_min, z_max = deformed_tet.bounds
    offsets_applied = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])
    deformed_tet.points -= offsets_applied

    deformed_tet.extract_surface().save(f'output_models/{model_name}_deformed_tet.stl')

    with open(f'pickle_files/deformed_{model_name}.pkl', 'wb') as f:
        pickle.dump(deformed_tet, f)

    print("Slicing with Cura...")
    abs_custom= os.path.abspath('config/core.def.json')
    abs_extruder = os.path.abspath('config/fdmextruder.def.json')
    abs_printer = os.path.abspath('config/fdmprinter.def.json')
    abs_model= os.path.abspath(f'output_models/{model_name}_deformed_tet.stl')
    abs_output = os.path.abspath(f'input_gcode/{model_name}_deformed_tet.gcode')

    command = [cura_path, 'slice', 
               '-j', abs_printer, 
               '-j', abs_extruder, 
               '-j', abs_custom]
    command = get_slicer_settings(command)
    command.extend(['-l', abs_model, '-o', abs_output])
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    deformed_tet = pickle.load(open(f'pickle_files/deformed_{model_name}.pkl', 'rb'))

    def tetrahedron_volume(p1, p2, p3, p4):
        mat = np.vstack([p2 - p1, p3 - p1, p4 - p1])
        return np.abs(np.linalg.det(mat)) / 6

    # Opt: Vectorized Barycentric Mapping Coordinates via LinAlg Matrix Solver
    def calc_barycentric_coordinates(tet_a, tet_b, tet_c, tet_d, point):
        T = np.column_stack((tet_a - tet_d, tet_b - tet_d, tet_c - tet_d))
        try:
            lambdas = np.linalg.solve(T, point - tet_d)
            return np.array([lambdas[0], lambdas[1], lambdas[2], 1.0 - np.sum(lambdas)])
        except np.linalg.LinAlgError:
            return np.array([0, 0, 0, 0])

    def project_point_onto_plane(plane_x_axis, plane_y_axis, point):
        projected_x = np.sum(plane_x_axis * point, axis=1)
        projected_y = np.sum(plane_y_axis * point, axis=1)
        return np.array([projected_x, projected_y]).T

    deformed_tet, _, _ = calculate_tet_attributes(deformed_tet)
    from pygcode import Line

    SEG_SIZE = 0.6 
    MAX_ROTATION = 30 
    MIN_ROTATION = -130 
    NOZZLE_OFFSET = 42 

    vertex_transformations = deformed_tet.points - input_tet.points
    tangential_vectors = np.cross( np.array([0, 0, 1]), input_tet.cell_data["cell_center"][:, :2])
    tangential_vectors /= np.linalg.norm(tangential_vectors, axis=1)[:, None]
    tangential_vectors[np.isnan(tangential_vectors).any(axis=1)] = [1, 0, 0]

    # OPT: Replace Python loop with np.bincount — O(n) numpy instead of O(n) Python.
    _cells_flat = input_tet.field_data["cells"].ravel()
    num_cells_per_vertex = np.bincount(_cells_flat, minlength=input_tet.number_of_points).astype(float)

    # OPT: Vectorized vertex rotation computation — batch SVD over all cells at once.
    # Replaces a Python for-loop with per-cell SVD, covariance, and scatter-add.
    _cells_arr   = deformed_tet.field_data["cells"]                       # (n_cells, 4)
    _n_cells_d   = deformed_tet.number_of_cells

    _new_verts   = deformed_tet.field_data["cell_vertices"][_cells_arr]   # (n_cells, 4, 3)
    _old_verts   = input_tet.field_data["cell_vertices"][_cells_arr]      # (n_cells, 4, 3)
    _new_cc      = deformed_tet.cell_data["cell_center"]                   # (n_cells, 3)
    _old_cc      = input_tet.cell_data["cell_center"]                      # (n_cells, 3)

    # Centre vertices around their cell centroid
    _new_verts  -= _new_cc[:, None, :]
    _old_verts  -= _old_cc[:, None, :]

    # Build per-cell plane_x_vector (normalised radial direction in XY, z=0)
    _xy_norm     = np.linalg.norm(_old_cc[:, :2], axis=1, keepdims=True) + 1e-8
    _plane_x     = np.concatenate([_old_cc[:, :2] / _xy_norm,
                                   np.zeros((_n_cells_d, 1))], axis=1)   # (n_cells, 3)
    # plane_y = [0, 0, 1] for all cells

    # Project vertices onto (plane_x, plane_y=Z-axis) — shape (n_cells, 4, 2)
    _new_px      = np.einsum('ni,nki->nk', _plane_x, _new_verts)         # dot with plane_x
    _new_py      = _new_verts[:, :, 2]                                    # dot with [0,0,1]
    _new_proj    = np.stack([_new_px, _new_py], axis=-1)

    _old_px      = np.einsum('ni,nki->nk', _plane_x, _old_verts)
    _old_py      = _old_verts[:, :, 2]
    _old_proj    = np.stack([_old_px, _old_py], axis=-1)

    # Batch covariance (n_cells, 2, 2) and SVD → rotation matrices
    _cov         = np.einsum('nki,nkj->nij', _new_proj, _old_proj)
    _U, _, _Vt   = np.linalg.svd(_cov)
    _rot_mats    = np.einsum('nij,njk->nik', _U, _Vt)                    # (n_cells, 2, 2)

    # Extract signed rotation angles from 2×2 rotation matrices
    _cos_val     = np.clip(_rot_mats[:, 0, 0], -1.0, 1.0)
    cell_rotations = -np.arccos(_cos_val)
    cell_rotations[_rot_mats[:, 1, 0] < 0] *= -1
    cell_rotations = np.clip(cell_rotations,
                             np.deg2rad(MIN_ROTATION), np.deg2rad(MAX_ROTATION))

    # Scatter-add per-vertex average rotation using np.add.at
    vertex_rotations = np.zeros(deformed_tet.number_of_points)
    _contrib = cell_rotations[:, None] / num_cells_per_vertex[_cells_arr] # (n_cells, 4)
    np.add.at(vertex_rotations, _cells_arr, _contrib)

    tet_rotation_matrices = calculate_rotation_matrices(input_tet, cell_rotations)

    # OPT: Vectorized z_squish_scales — batch tetrahedron volume via np.linalg.det.
    # Replaces a Python loop with per-cell volume calculations.
    def _batch_tet_volume(verts4):
        """verts4: (n, 4, 3) → unsigned tet volumes (n,)"""
        d = verts4[:, 1:] - verts4[:, :1]   # (n, 3, 3)
        return np.abs(np.linalg.det(d)) / 6.0

    _warp_v    = deformed_tet.field_data["cell_vertices"][_cells_arr]     # (n_cells, 4, 3)
    _unwarp_v  = input_tet.field_data["cell_vertices"][_cells_arr]        # (n_cells, 4, 3)
    _vol_unwarp = _batch_tet_volume(_unwarp_v)
    _vol_warp   = _batch_tet_volume(_warp_v)
    z_squish_scales = _vol_unwarp / (_vol_warp + 1e-15)

    pos = np.array([0., 0., 20.])
    feed = 5000
    gcode_points = []
    with open(f'input_gcode/{model_name}_deformed_tet.gcode', 'r') as fh:
        for line_text in fh.readlines():
            line = Line(line_text)

            if not line.block.gcodes:
                continue

            for gcode in sorted(line.block.gcodes):
                if gcode.word == "G01" or gcode.word == "G00":
                    prev_pos = pos.copy()

                    if gcode.X is not None:
                        pos[0] = gcode.X
                    if gcode.Y is not None:
                        pos[1] = gcode.Y
                    if gcode.Z is not None:
                        pos[2] = gcode.Z

                    inv_time_feed = None
                    for word in line.block.words:
                        if word.letter == "F":
                            feed = word.value

                    extrusion = None
                    for param in line.block.modal_params:
                        if param.letter == "E":
                            extrusion = param.value

                    delta_pos = pos - prev_pos
                    distance = np.linalg.norm(delta_pos)
                    if distance > 0:
                        num_segments = -(-distance // SEG_SIZE) 
                        seg_distance = distance/num_segments

                        time_to_complete_move = (1/feed) * seg_distance 
                        if time_to_complete_move == 0:
                            inv_time_feed = None
                        else:
                            inv_time_feed = 1/time_to_complete_move 

                        for i in range(int(num_segments)):
                            gcode_points.append({
                                "position": (prev_pos + delta_pos * (i+1) / num_segments),
                                "command": gcode.word,
                                "extrusion": extrusion/num_segments if extrusion is not None else None,
                                "inv_time_feed": inv_time_feed,
                                "move_length": seg_distance,
                                "start_position": prev_pos,
                                "end_position": pos,
                                "unsegmented_move_length": distance,
                                "after_retract": False,
                                "feed": feed
                            })
                    else:
                        time_to_complete_move = (1/feed) * distance
                        if time_to_complete_move == 0:
                            inv_time_feed = None
                        else:
                            inv_time_feed = 1/time_to_complete_move 

                        gcode_points.append({
                            "position": pos.copy(),
                            "command": gcode.word,
                            "extrusion": extrusion,
                            "inv_time_feed": inv_time_feed,
                            "move_length": distance,
                            "unsegmented_move_length": distance,
                            "after_retract": False,
                            "feed": feed
                        })

    gcode_points_containing_cells = deformed_tet.find_containing_cell([point["position"] for point in gcode_points])
    gcode_points_closest_cells = deformed_tet.find_closest_cell([point["position"] for point in gcode_points])

    # OPT: Pre-compute (new_position, rotation) for every gcode point using batched linear solves.
    # The original called np.linalg.solve once per point inside the sequential loop (40s).
    # Here we solve all points at once with np.linalg.solve on a batched (n, 3, 3) system,
    # then the sequential loop only handles smoothing / travelling logic using pre-computed values.
    print("Pre-computing barycentric transforms (batch)...")
    _n_pts = len(gcode_points)
    _positions_all = np.array([p["position"] for p in gcode_points])     # (n_pts, 3)
    _commands_all  = [p["command"] for p in gcode_points]
    _containing    = np.array(gcode_points_containing_cells, dtype=np.int32)
    _closest       = np.array(gcode_points_closest_cells,   dtype=np.int32)
    _is_g01        = np.array([c == "G01" for c in _commands_all])

    # Determine effective cell per point (mirrors barycentric_interpolate logic):
    #  G00 + containing == -1  → eff = -1  (will produce None result)
    #  G01 + containing == -1  → eff = closest cell
    #  otherwise               → eff = containing cell
    _eff_cells = _containing.copy()
    _fallback  = _is_g01 & (_containing == -1)
    _eff_cells[_fallback] = _closest[_fallback]

    _valid = _eff_cells >= 0                                               # points we can solve
    _valid_idx = np.where(_valid)[0]

    _cells_data = deformed_tet.field_data["cells"]
    _verts_data = deformed_tet.field_data["cell_vertices"]

    _vc           = _eff_cells[_valid_idx]
    _vert_indices = _cells_data[_vc]                                       # (n_valid, 4)
    _cell_verts   = _verts_data[_vert_indices]                             # (n_valid, 4, 3)

    # Build batched T matrices: each column of T[i] is (vert_j - vert_d) for j in {a,b,c}
    _tet_d = _cell_verts[:, 3]                                             # (n_valid, 3)
    _T     = (_cell_verts[:, :3] - _cell_verts[:, 3:4]).transpose(0, 2, 1)  # (n_valid, 3, 3)
    _rhs   = _positions_all[_valid_idx] - _tet_d                           # (n_valid, 3)

    # Batch solve — numpy's gufunc signature is (m,m),(m,n)->(m,n), so b must be
    # (..., m, n), NOT (..., m).  Adding a trailing K=1 column makes n=1 explicit;
    # squeeze removes it afterward.  Without this, numpy reads the batch axis as
    # the core "m" dimension and raises a shape-mismatch ValueError.
    try:
        _lambdas = np.linalg.solve(_T, _rhs[:, :, None]).squeeze(-1)      # (n_valid, 3)
    except np.linalg.LinAlgError:
        # Fallback: solve element-wise when any matrix is singular
        _lambdas = np.zeros((len(_valid_idx), 3))
        for _k in range(len(_valid_idx)):
            try:
                _lambdas[_k] = np.linalg.solve(_T[_k], _rhs[_k])
            except np.linalg.LinAlgError:
                _lambdas[_k] = [0.0, 0.0, 0.0]

    _bary      = np.empty((len(_valid_idx), 4))
    _bary[:, :3] = _lambdas
    _bary[:, 3]  = 1.0 - _lambdas.sum(axis=1)
    _bary_ok   = _bary.sum(axis=1) <= 1.01                                # mirrors `> 1.01` → None

    # Compute pre-transformed positions and rotations
    _vt        = vertex_transformations[_vert_indices]                     # (n_valid, 4, 3)
    _pre_pos   = _positions_all[_valid_idx] - (_vt * _bary[:, :, None]).sum(axis=1)
    _pre_pos[~_bary_ok] = np.nan

    _vr        = vertex_rotations[_vert_indices]                           # (n_valid, 4)
    _pre_rot   = (_vr * _bary).sum(axis=1)
    _pre_rot[~_bary_ok] = np.nan

    # Map results back to all-point arrays (NaN = None in original code)
    pre_new_positions = np.full((_n_pts, 3), np.nan)
    pre_rotations     = np.full(_n_pts,      np.nan)
    pre_new_positions[_valid_idx] = _pre_pos
    pre_rotations[_valid_idx]     = _pre_rot

    # OPT: Pre-compute validity as a boolean array — avoids calling np.any(np.isnan(...))
    # (which allocates a small array) on every single iteration of the reforming loop.
    _pre_valid = ~np.any(np.isnan(pre_new_positions), axis=1)  # (n_pts,) bool

    print("Reforming...")
    prev_time = time.time()
    new_gcode_points = []
    prev_new_position = None
    travelling_over_air = False
    travelling = False
    prev_position = None
    prev_rotation = 0
    prev_travelling = False
    prev_command = "G00"
    ROTATION_AVERAGING_ALPHA = 0.2
    RETRACTION_LENGTH = 1.0
    ROTATION_MAX_DELTA = np.deg2rad(1)
    MAX_EXTRUSION_MULTIPLIER = 10
    lost_vertices = []
    highest_printed_point = 0
    for cell_index, (gcode_point, containing_cell_index) in enumerate(zip(gcode_points, gcode_points_containing_cells)):
        position = gcode_point["position"]
        command = gcode_point["command"]
        inv_time_feed = gcode_point["inv_time_feed"]
        extrusion = gcode_point["extrusion"]

        # OPT: Replace per-point barycentric solve with pre-computed lookup.
        _pnp = pre_new_positions[cell_index]
        _pr  = pre_rotations[cell_index]
        _pos_valid = bool(_pre_valid[cell_index])      # O(1) boolean array lookup

        dont_smooth_rotation = False
        if _pos_valid:
            new_position = _pnp.copy()
            rotation     = float(_pr)
        else:
            new_position = None
            rotation     = None
        if new_position is None:
            if command == "G01":
                lost_vertices.append(position)
                continue
            elif command == "G00" and not travelling_over_air and prev_new_position is not None:
                new_position = np.array([prev_new_position[0], prev_new_position[1], highest_printed_point]) 
                rotation = max(min(prev_rotation, np.deg2rad(45)), np.deg2rad(-45))
                dont_smooth_rotation = True 
                travelling_over_air = True
            elif travelling_over_air:
                continue
            else:
                continue
        else:
            if travelling_over_air:
                new_position[2] = highest_printed_point 
                rotation = max(min(rotation, np.deg2rad(45)), np.deg2rad(-45)) 
                dont_smooth_rotation = True 
            travelling_over_air = False

        extrusion_multiplier = 1
        if extrusion is not None and extrusion != RETRACTION_LENGTH and extrusion != -RETRACTION_LENGTH:
            extrusion_multiplier = extrusion_multiplier * z_squish_scales[containing_cell_index]
            extrusion = extrusion * min(extrusion_multiplier, MAX_EXTRUSION_MULTIPLIER)
        elif extrusion == -RETRACTION_LENGTH:
            travelling = True
        elif extrusion == RETRACTION_LENGTH:
            travelling = False
        if prev_rotation is not None and not dont_smooth_rotation:
            rotation = ROTATION_AVERAGING_ALPHA * rotation + (1 - ROTATION_AVERAGING_ALPHA) * prev_rotation

        if prev_rotation is not None and prev_new_position is not None and np.abs(rotation - prev_rotation) > ROTATION_MAX_DELTA:
            delta_rotation = rotation - prev_rotation
            num_interpolations = int(np.abs(delta_rotation) / ROTATION_MAX_DELTA) + 1
            delta_pos = new_position - prev_new_position
            for i in range(num_interpolations):
                new_gcode_points.append({
                    "position": prev_new_position + (delta_pos * ((i+1) / num_interpolations)),
                    "original_position": position,
                    "rotation": prev_rotation + (delta_rotation * ((i+1) / num_interpolations)),
                    "command": prev_command,
                    "extrusion": extrusion/num_interpolations if extrusion is not None else None,
                    "inv_time_feed": inv_time_feed * num_interpolations if inv_time_feed is not None else None,
                    "extrusion_multiplier": extrusion_multiplier,
                    "feed": gcode_point["feed"],
                    "travelling": prev_travelling
                })
        else:
            new_gcode_points.append({
                "position": new_position,
                "original_position": position,
                "rotation": rotation,
                "command": command,
                "extrusion": extrusion,
                "inv_time_feed": inv_time_feed,
                "extrusion_multiplier": extrusion_multiplier,
                "feed": gcode_point["feed"],
                "travelling": travelling
            })

        prev_rotation = rotation
        prev_new_position = new_position.copy()
        prev_travelling = travelling
        prev_command = command

        if command == "G01" and extrusion is not None and extrusion > 0 and (highest_printed_point != 0 or new_position[2] < 1):
            highest_printed_point = max(highest_printed_point, new_position[2])

    print(f"Lost {len(lost_vertices)} vertices")
    print(f"Reforming done: {round(time.time()- prev_time, 2)}s")
    prev_time = time.time()

    prev_r = 0
    prev_theta = 0
    prev_z = 20
    theta_accum = 0

    with open(f'output_gcode/{model_name}.gcode', 'w') as fh:
        print(f"Saving to output_gcode/{model_name}.gcode")
        fh.write("G94 ; mm/min feed  \n")
        fh.write("G28 ; home \n")
        fh.write("M83 ; relative extrusion \n")
        fh.write("G1 E10 ; prime extruder \n")
        fh.write("G94 ; mm/min feed \n")
        fh.write("G90 ; absolute positioning \n")
        fh.write(f"G0 C{prev_theta} X{prev_r} Z{prev_z} B0 ; go to start \n")
        fh.write("G93 ; inverse time feed \n")

        for i, point in enumerate(new_gcode_points):
            position = point["position"]
            rotation = point["rotation"]

            if np.all(np.isnan(position)):
                continue

            if position[2] < 0:
                continue

            z_hop = 0
            if point["travelling"]:
                z_hop = 1

            r = np.linalg.norm(position[:2])
            theta = np.arctan2(position[1], position[0])
            z = position[2]

            r += -np.sin(rotation) * (NOZZLE_OFFSET + z_hop)
            z += (np.cos(rotation) - 1) * (NOZZLE_OFFSET + z_hop) + z_hop

            delta_theta = theta - prev_theta
            if delta_theta > np.pi:
                delta_theta -= 2*np.pi
            if delta_theta < -np.pi:
                delta_theta += 2*np.pi

            theta_accum += delta_theta

            string = f"{point['command']} C{np.rad2deg(theta_accum):.5f} X{r:.5f} Z{z:.5f} B{np.rad2deg(rotation):.5f}"

            if point["extrusion"] is not None:
                string += f" E{point['extrusion']:.4f}"

            no_feed_value = False
            if point["inv_time_feed"] is not None:
                string += f" F{(point['inv_time_feed']):.4f}"
            else:
                string += f" F20000"
                fh.write(f"G94\n")
                no_feed_value = True

            fh.write(string + "\n")

            if no_feed_value:
                fh.write(f"G93\n") 

            prev_r = r
            prev_theta = theta
            prev_z = z
    print(f'Total time: {time.time()-total_t}')