import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from utils import *
from solution import *

# --- 1. FONCTIONS UTILITAIRES ---
def path_length(path):
    if not path: return 0
    coords = []
    for item in path:
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (tuple, list)):
            coords.append(item[1])
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], (int, float, np.integer)):
            coords.append(item)
    if len(coords) < 2: return 0
    pts = np.array(coords)
    return np.sum(np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1)))

def extract_flux_path(V, node_to_idx, graph, start, goal):
    path = [start]
    curr = start
    visited = {start}
    for _ in range(len(graph) * 2):
        if curr == goal: break
        if curr not in graph: break
        neighbors = list(graph[curr])
        if not neighbors: break
        next_node = min(neighbors, key=lambda v: V[node_to_idx[v]])
        if next_node in visited: break
        path.append(next_node)
        visited.add(next_node)
        curr = next_node
    return path

# --- 2. SOLVEURS ---

def solve_flux_spectral(graph, etat):
    # --- PHASE 1 : CONSTRUCTION (Hors Chrono) ---
    nodes = list(graph.keys())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    data, rows, cols = [], [], []
    for u, neighbors in graph.items():
        if u not in node_to_idx: continue
        u_idx = node_to_idx[u]
        degree = 0
        for v in neighbors:
            if v in node_to_idx:
                v_idx = node_to_idx[v]
                data.append(-1)
                rows.append(u_idx)
                cols.append(v_idx)
                degree += 1
        data.append(degree)
        rows.append(u_idx)
        cols.append(u_idx)

    L = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    b = np.zeros(num_nodes)
    
    try:
        b[node_to_idx[etat['start']]] = 1
        b[node_to_idx[etat['goal']]] = -1
    except KeyError:
        return 0, 0, []

    keep = [i for i in range(num_nodes) if i != node_to_idx[etat['goal']]]
    V = np.zeros(num_nodes)
    
    # --- PHASE 2 : RÉSOLUTION (Chrono Activé) ---
    # On mesure uniquement le temps pour trouver le chemin une fois le graphe connu
    t0 = time.time()
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=MatrixRankWarning)
            V[keep] = spsolve(L[keep, :][:, keep], b[keep])
        
        # L'extraction fait partie de la "résolution" du chemin
        path = extract_flux_path(V, node_to_idx, graph, etat['start'], etat['goal'])
        
    except:
        return 0, 0, []

    dt = time.time() - t0
    length = path_length(path)
    return dt, length, path

def solve_astar_optimal(graph, etat):
    # Pour A*, tout est "résolution" car il n'y a pas de pré-calcul matriciel
    t0 = time.time()
    try:
        res = A_star(graph, (etat['start'], etat['orientation']), etat['goal'])
        path = []
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], list):
            path = res[0]
        elif isinstance(res, list):
            path = res
    except Exception:
        path = []
    dt = time.time() - t0
    length = path_length(path)
    return dt, length, path

# --- 3. BENCHMARK ULTIME ---

def run_ultimate_benchmark():
    M, N = 50, 50
    obstacle_counts = range(10, 301, 40)
    
    SAMPLES_PER_DENSITY = 10
    RUNS_PER_SAMPLE = 3
    
    results = {
        'obstacles': [],
        'time_flux': [],
        'time_astar': [],
        'dist_flux': [],
        'dist_astar': []
    }

    print(f"{'NB OBS':<8} | {'T_FLUX (Avg)':<12} | {'T_ASTAR (Avg)':<12} | {'SAMPLES':<8}")
    print("-" * 60)

    for nb_obs in obstacle_counts:
        medians_flux_time = []
        medians_astar_time = []
        samples_flux_dist = []
        samples_astar_dist = []
        
        valid_samples = 0
        attempts = 0
        
        while valid_samples < SAMPLES_PER_DENSITY and attempts < 60:
            attempts += 1
            try:
                instance = genere_instance(M, N, nb_obstacles=nb_obs, min_path_len=15)
                etat = {
                    'start': instance['start'],
                    'goal': instance['goal'],
                    'orientation': instance['orientation']
                }
                graph = instance['graph']
                
                _, d_f_check, _ = solve_flux_spectral(graph, etat)
                if d_f_check == 0: continue
                _, d_a_check, _ = solve_astar_optimal(graph, etat)
                if d_a_check == 0: continue
                
                local_flux_times = []
                for _ in range(RUNS_PER_SAMPLE):
                    dt, _, _ = solve_flux_spectral(graph, etat)
                    local_flux_times.append(dt)
                median_t_flux = np.median(local_flux_times)
                
                local_astar_times = []
                for _ in range(RUNS_PER_SAMPLE):
                    dt, _, _ = solve_astar_optimal(graph, etat)
                    local_astar_times.append(dt)
                median_t_astar = np.median(local_astar_times)
                
                medians_flux_time.append(median_t_flux)
                medians_astar_time.append(median_t_astar)
                samples_flux_dist.append(d_f_check)
                samples_astar_dist.append(d_a_check)
                
                valid_samples += 1
                
            except Exception:
                continue
        
        if valid_samples > 0:
            avg_t_flux = np.mean(medians_flux_time)
            avg_t_astar = np.mean(medians_astar_time)
            avg_d_flux = np.mean(samples_flux_dist)
            avg_d_astar = np.mean(samples_astar_dist)
            
            results['obstacles'].append(nb_obs)
            results['time_flux'].append(avg_t_flux)
            results['time_astar'].append(avg_t_astar)
            results['dist_flux'].append(avg_d_flux)
            results['dist_astar'].append(avg_d_astar)
            
            print(f"{nb_obs:<8} | {avg_t_flux:<12.4f} | {avg_t_astar:<12.4f} | {valid_samples:<8}")
        else:
            print(f"{nb_obs:<8} | Trop dense (0 valide)")

    return results

def plot_benchmark(res):
    if not res['obstacles']: return

    x = res['obstacles']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(x, res['time_astar'], '-o', label='A* (Optimal)', color='cyan', lw=2)
    ax1.plot(x, res['time_flux'], '-o', label='Flux Spectral (Résolution seule)', color='orange', lw=2)
    ax1.set_xlabel("Nombre d'obstacles")
    ax1.set_ylabel("Temps (s)")
    ax1.set_title(f"Performance Online (Construction exclue)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ratios = np.array(res['dist_flux']) / np.array(res['dist_astar'])
    ax2.plot(x, ratios, '-o', color='purple', lw=2)
    ax2.axhline(y=1.0, color='green', linestyle='--', label="Optimal (1.0)")
    ax2.set_xlabel("Nombre d'obstacles")
    ax2.set_ylabel("Ratio k")
    ax2.set_title("Coût de l'approximation")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Benchmark Ultime (Construction exclue du temps Flux) ...")
    data = run_ultimate_benchmark()
    plot_benchmark(data)
