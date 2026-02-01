"""
# python bench.py bench -n 100 -m 1000 -obs 0 100 300 6000 8000 12000 1 -i 8 -s all
# python bench.py solve input/input.txt -s Bfs_optimise || A_star
# python bench.py bench -n 300 600 1000 1600 3000 -m 1000 -obs 3000 -i 4 -s all

# python bench.py bench -n 10 20 30 40 50 -m 10 20 30 40 50 -obs 3 20 30 40 50 -i 10 -s all -t v
# python bench.py bench -n 20 -m 20  -obs 10 20 30 40 -i 10 -s all -t u
# python bench.py bench -i 6 -s all -t v -m 100 1000 100 -n 100 1000 100
"""


"""
Script de benchmark : Robot sur Grille.
Version : FINALE (Pro & Robust)

Ce script permet de :
1. Résoudre une instance unique et sauvegarder le chemin.
2. Lancer un benchmark massif en parallèle.
3. Générer des graphiques comparant la rapidité (Temps) et la qualité (Longueur du chemin).

Utilisation :
-------------
# 1. Mode Benchmark Combinatoire ('u') : N varie, M et Obs sont fixes.
python bench.py bench -n 20 40 60 80 -m 50 -obs 10 -i 10 -s all -t u

# 2. Mode Benchmark Linéaire ('v') : Tout varie ensemble (Triplets [N,M,Obs]).
python bench.py bench -n 20 40 -m 20 40 -obs 5 10 -i 10 -s all -t v

# 3. Mode Résolution simple :
python bench.py solve input/input.txt -s A_star_spectrale
"""

import time
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from itertools import product
import argparse
import sys
import csv
import os
import multiprocessing

# --- Imports locaux ---
# Ces fichiers doivent être dans le même répertoire
from solveurs import Bfs_optimise, A_star, h_super_spectrale, h_manhattan
from utils import load, genere_instance


# =============================================================================
# WRAPPER MULTIPROCESSING (CRUCIAL)
# =============================================================================

class SolverWrapper:
    """
    Remplace functools.partial pour le multiprocessing.
    
    Problème résolu : 'partial' ne conserve pas l'attribut __name__ lors du
    processus de sérialisation (pickling) vers les workers.
    Cette classe encapsule la fonction et ses arguments (ex: l'heuristique h)
    tout en garantissant que .__name__ est toujours accessible.
    """
    def __init__(self, func, name, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.__name__ = name

    def __call__(self, *args, **kwargs):
        # Fusionne les arguments par défaut (self.kwargs) avec ceux de l'appel
        combined_kwargs = {**self.kwargs, **kwargs}
        return self.func(*args, **combined_kwargs)


# =============================================================================
# CONFIGURATION DES SOLVEURS
# =============================================================================

SOLVER_MAP = {
    'Bfs_optimise': Bfs_optimise,
    
    # On crée des variantes d'A* pré-configurées avec SolverWrapper
    'A_star_manhattan': SolverWrapper(A_star, 'A_star_manhattan', h=h_manhattan),
    'A_star_spectrale': SolverWrapper(A_star, 'A_star_spectrale', h=h_super_spectrale),
}


# =============================================================================
# FONCTIONS CŒUR (CORE)
# =============================================================================

def run_solver(filename, solveur, output):
    """
    Mode 'solve' : Résout un fichier unique, affiche les stats et sauve le chemin.
    """
    print(f"--- Mode Résolution : {filename} ---")
    
    # Chargement
    G, etat = load(filename)
    if G is None or etat is None:
        print(f"Erreur: Impossible de charger '{filename}'.")
        return

    print(f"Lancement de : {solveur.__name__}...")
    
    # Exécution
    t0 = time.process_time()
    path = solveur(G, (etat['start'], etat['orientation']), etat['goal'])
    duration = time.process_time() - t0

    # Résultat
    if path:
        cout = len(path) - 1
        print(f"-> Succès !")
        print(f"   Temps d'exécution : {duration:.6f} s")
        print(f"   Longueur du chemin : {cout} pas")
        
        output_path = os.path.join(output, f"chemin_{os.path.basename(filename)}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for etape in path:
                f.write(f"{etape}\n")
        print(f"   Chemin sauvegardé dans : '{output_path}'")
    else:
        print("--- Échec : Aucun chemin trouvé. ---")


def worker_task(args):
    """
    Tâche exécutée par un processus (Worker) du pool.
    1. Génère une instance aléatoire.
    2. Lance TOUS les solveurs demandés sur cette instance.
    3. Retourne les métriques (Temps et Coût) pour chaque solveur.
    """
    N, M, obs_count, solveurs = args
    results = {}

    try:
        # 1. Génération de l'instance
        inst = genere_instance(M, N, obs_count)
        G = inst['graph']
        start = (inst['start'], inst['orientation'])
        goal = inst['goal']

        # 2. Contexte (Pré-calculs éventuels pour heuristiques complexes)
        context_data = {
            'psi': inst.get('psi'),
            'eigenvalue': inst.get('eigenvalue'), 
            'N_cols': N - 1
        }

        # 3. Exécution des solveurs
        for solver_obj in solveurs:
            name = solver_obj.__name__
            
            t0 = time.process_time()
            path = solver_obj(G, start, goal, **context_data)
            t1 = time.process_time()
            
            duration = t1 - t0
            
            # Si path est None (échec), le coût est None
            cost = len(path) - 1 if path is not None else None
            
            results[name] = {'time': duration, 'cost': cost}

        return (N, M, obs_count), results, None 

    except Exception as e:
        # On capture l'erreur ici pour ne pas crasher tout le benchmark
        return (N, M, obs_count), None, str(e)


def run_benchmark(sizes_n, sizes_m, obs_list, mode, n_inst, solveurs, output, workers):
    """
    Orchestre le benchmark en parallèle.
    """
    print(f"--- Mode Benchmark (Workers: {workers}) ---")
    
    # 1. Préparation des tâches
    tasks = []
    
    if mode == "u":
        # Produit Cartésien
        combinations = list(product(sizes_n, sizes_m, obs_list))
        print(f"Mode 'u' (Combinatoire) : {len(combinations)} configurations.")
    elif mode == "v":
        # Zip (Linéaire)
        if not (len(sizes_n) == len(sizes_m) == len(obs_list)):
            raise ValueError("Mode 'v' : Les listes N, M et Obs doivent avoir la même taille.")
        combinations = list(zip(sizes_n, sizes_m, obs_list))
        print(f"Mode 'v' (Linéaire) : {len(combinations)} configurations.")
    else:
        raise ValueError("Mode inconnu (utilisez 'u' ou 'v').")

    # On multiplie par le nombre d'instances par config
    for params in combinations:
        for _ in range(n_inst):
            tasks.append((*params, solveurs))

    total_tasks = len(tasks)
    print(f"Lancement de {total_tasks} simulations...")
    
    # Structure de stockage : raw_data[(solver, N, M, O)] = {'times': [], 'costs': []}
    raw_data = defaultdict(lambda: {'times': [], 'costs': []})

    # 2. Exécution Multiprocessing
    with multiprocessing.Pool(processes=workers) as pool:
        # imap_unordered permet d'avoir une barre de progression fluide
        iterator = pool.imap_unordered(worker_task, tasks)

        for i, (params, task_res, error) in enumerate(iterator, 1):
            if error:
                print(f" [Erreur] {params} : {error}")
                continue
            
            # Agrégation des résultats
            for s_name, metrics in task_res.items():
                key = (s_name, *params)
                raw_data[key]['times'].append(metrics['time'])
                
                # On ne stocke le coût que si le solveur a réussi
                if metrics['cost'] is not None:
                    raw_data[key]['costs'].append(metrics['cost'])

            # Barre de progression simple
            if i % max(1, total_tasks//10) == 0 or i == total_tasks:
                print(f"  Progression : {int(i/total_tasks*100)}%")

    # 3. Analyse et Sauvegarde
    out_dir = os.path.join(output, "bench")
    csv_path = os.path.join(out_dir, "resultats_complets.csv")
    
    stats_processed = analyser_et_sauvegarder(raw_data, csv_path)

    # 4. Traçage des courbes
    solver_names = [s.__name__ for s in solveurs]
    tracer_courbes_completes(stats_processed, solver_names, sizes_n, sizes_m, obs_list, out_dir, mode)


# =============================================================================
# ANALYSE STATISTIQUE ET CSV
# =============================================================================

def analyser_et_sauvegarder(raw_data, filepath):
    """
    Calcule les statistiques (Médiane, Moyenne, Ecart-Type) pour le Temps et le Coût.
    Écrit le tout dans un CSV détaillé.
    """
    print(f"\n[Analyse] Écriture des résultats -> {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    stats_processed = {}

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "solver", "N", "M", "Obs", 
            "time_median", "time_mean", "time_std", 
            "cost_median", "cost_mean", "cost_std", 
            "success_count"
        ])

        # Tri des clés pour un fichier propre
        sorted_keys = sorted(raw_data.keys(), key=lambda x: (x[0], x[1], x[2], x[3]))

        for key in sorted_keys:
            d = raw_data[key]
            times = np.array(d['times'], dtype=float)
            costs = np.array(d['costs'], dtype=float)
            
            # Stats Temps
            if times.size > 0:
                t_med = np.median(times)
                t_mean = np.mean(times)
                t_std = np.std(times)
            else:
                t_med = t_mean = t_std = 0

            # Stats Coût
            if costs.size > 0:
                c_med = np.median(costs)
                c_mean = np.mean(costs)
                c_std = np.std(costs)
            else:
                c_med = c_mean = c_std = 0

            # Stockage structuré pour le plot
            stats_processed[key] = {
                'time': {'median': t_med, 'std': t_std},
                'cost': {'median': c_med, 'std': c_std}
            }
            
            s_name, n, m, o = key
            writer.writerow([
                s_name, n, m, o, 
                f"{t_med:.6f}", f"{t_mean:.6f}", f"{t_std:.6f}", 
                f"{c_med:.2f}", f"{c_mean:.2f}", f"{c_std:.2f}", 
                costs.size
            ])
            
    return stats_processed


# =============================================================================
# GRAPHIQUES PRO (PLOTTING)
# =============================================================================

def tracer_courbes_completes(stats, solver_names, Ns, Ms, Os, plot_dir, mode):
    """
    Génère les graphiques pour :
    1. La performance Temporelle (Médiane).
    2. La qualité de la solution (Longueur Médiane).
    
    Inclus des échelles Linéaires et Logarithmiques.
    """
    # Force 'Agg' pour éviter les erreurs d'affichage (pas de GUI requis)
    matplotlib.use('Agg') 
    
    # Style moderne
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot') # Fallback

    os.makedirs(os.path.join(plot_dir, "plots"), exist_ok=True)
    plot_dir = os.path.join(plot_dir, "plots")

    # --- 1. Détection de la variable d'axe X ---
    if mode == "u":
        # On cherche quelle variable change parmi N, M, Obs
        if len(Ns) > 1:   
            var_name, x_vals, x_label = 'N', sorted(Ns), "Taille N (Lignes)"
            get_key = lambda s, x: (s, x, Ms[0], Os[0])
        elif len(Ms) > 1: 
            var_name, x_vals, x_label = 'M', sorted(Ms), "Taille M (Colonnes)"
            get_key = lambda s, x: (s, Ns[0], x, Os[0])
        else:             
            var_name, x_vals, x_label = 'Obs', sorted(Os), "Nombre d'Obstacles"
            get_key = lambda s, x: (s, Ns[0], Ms[0], x)
            
    else: # mode v
        var_name, x_vals, x_label = 'Config', list(range(len(Ns))), "Configuration (N, M, O)"
        # x est un index ici
        get_key = lambda s, x_idx: (s, Ns[x_idx], Ms[x_idx], Os[x_idx])

    # --- 2. Définition des Métriques à tracer ---
    metrics_to_plot = [
        {
            'id': 'time', 
            'title': 'Temps d\'exécution', 
            'ylabel': 'Temps Médian (s)', 
            'log_possible': True
        },
        {
            'id': 'cost', 
            'title': 'Longueur du Chemin (Coût)', 
            'ylabel': 'Longueur Médiane (pas)', 
            'log_possible': False # Le log sur le coût est rarement pertinent
        }
    ]
    
    # Couleurs fixes par solveur pour cohérence
    colors = plt.cm.tab10(np.linspace(0, 1, len(solver_names)))
    c_map = dict(zip(solver_names, colors))

    print(f"[Plot] Génération des courbes (Variable: {var_name})...")

    # --- 3. Boucle de génération ---
    for met in metrics_to_plot:
        
        # On génère Linear ET Log (si pertinent)
        scales = ["linear"]
        if met['log_possible']: scales.append("log")

        for scale in scales:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            has_data = False
            for solv in solver_names:
                medians, stds, xs = [], [], []
                
                for x in x_vals:
                    k = get_key(solv, x)
                    if k in stats:
                        # On récupère la médiane et le std
                        val_med = stats[k][met['id']]['median']
                        val_std = stats[k][met['id']]['std']
                        
                        # On ignore si la valeur est 0 (ex: échec total)
                        if val_med > 0: 
                            medians.append(val_med)
                            stds.append(val_std)
                            xs.append(x)
                
                if xs:
                    has_data = True
                    medians, stds = np.array(medians), np.array(stds)
                    col = c_map[solv]
                    
                    # 1. Courbe Principale (MÉDIANE)
                    ax.plot(xs, medians, 'o-', lw=2, label=solv, color=col)
                    
                    # 2. Zone d'ombre (MÉDIANE +/- STD)
                    lower = np.maximum(0, medians - stds)
                    upper = medians + stds
                    ax.fill_between(xs, lower, upper, color=col, alpha=0.15)

            if not has_data:
                plt.close(fig)
                continue

            # Cosmétique du graphe
            ax.set_title(f"{met['title']} vs {var_name} ({scale.capitalize()})", fontsize=14, fontweight='bold')
            ax.set_ylabel(met['ylabel'], fontsize=12)
            ax.set_xlabel(x_label, fontsize=12)
            ax.legend(frameon=True, fancybox=True, framealpha=0.9, shadow=True, loc='best')
            ax.grid(True, which="both", linestyle='--', alpha=0.6)

            if scale == "log":
                ax.set_yscale('log')

            # Gestion des étiquettes pour le mode 'v'
            if mode == "v":
                ax.set_xticks(x_vals)
                labels = [f"{n}x{m}\n({o} obs)" for n, m, o in zip(Ns, Ms, Os)]
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

            plt.tight_layout()
            
            filename = f"Graph_{var_name}_{met['id']}_{scale}.png"
            plt.savefig(os.path.join(plot_dir, filename), dpi=200)
            plt.close(fig)
            print(f"  -> Sauvegardé : {filename}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark Robot Grille")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- Sous-commande : SOLVE ---
    p_solve = subparsers.add_parser('solve', help="Résoudre une instance unique")
    p_solve.add_argument('filename', help="Fichier d'entrée")
    p_solve.add_argument('-o', '--output', default='output')
    p_solve.add_argument('-s', '--solveur', default='A_star_manhattan', choices=SOLVER_MAP.keys())

    # --- Sous-commande : BENCH ---
    p_bench = subparsers.add_parser('bench', help="Lancer un benchmark")
    p_bench.add_argument('-n', '--sizes-n', nargs='+', type=int, help="Tailles N")
    p_bench.add_argument('-m', '--sizes-m', nargs='+', type=int, help="Tailles M")
    p_bench.add_argument('-obs', '--obstacles', nargs='+', type=int, help="Nombre d'obstacles")
    p_bench.add_argument('-i', '--instances', type=int, default=10, help="Nb répétitions par config")
    p_bench.add_argument('-s', '--solveurs', nargs='+', required=True, help="Liste solveurs ou 'all'")
    p_bench.add_argument('-t', '--test_type', default='u', choices=['u', 'v'], help="Mode u (produit) ou v (linéaire)")
    p_bench.add_argument('-o', '--output', default='output')
    p_bench.add_argument('-w', '--workers', type=int, default=multiprocessing.cpu_count()-1)

    args = parser.parse_args()

    # Dispatching
    if args.mode == 'solve':
        run_solver(args.filename, SOLVER_MAP[args.solveur], args.output)
    
    elif args.mode == 'bench':
        # Sélection des solveurs
        if 'all' in args.solveurs:
            selected_solvers = list(SOLVER_MAP.values())
        else:
            selected_solvers = []
            for s in args.solveurs:
                if s in SOLVER_MAP:
                    selected_solvers.append(SOLVER_MAP[s])
                else:
                    print(f"Attention: Solveur '{s}' inconnu.")
            if not selected_solvers:
                sys.exit("Erreur : Aucun solveur valide sélectionné.")
        
        run_benchmark(
            args.sizes_n, args.sizes_m, args.obstacles, 
            args.test_type, args.instances, selected_solvers, 
            args.output, args.workers
        )

if __name__ == "__main__":
    main()
