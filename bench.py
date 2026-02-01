"""
# python bench.py bench -n 100 -m 1000 -obs 0 100 300 6000 8000 12000 1 -i 8 -s all
# python bench.py solve input/input.txt -s Bfs_optimise || A_star
# python bench.py bench -n 300 600 1000 1600 3000 -m 1000 -obs 3000 -i 4 -s all

# python bench.py bench -n 10 20 30 40 50 -m 10 20 30 40 50 -obs 3 20 30 40 50 -i 10 -s all -t v
# python bench.py bench -n 20 -m 20  -obs 10 20 30 40 -i 10 -s all -t u
# python bench.py bench -i 6 -s all -t v -m 100 1000 100 -n 100 1000 100
"""

import time
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from itertools import product
import argparse
import sys
import csv
import os
import multiprocessing

from solveurs import Bfs_optimise, A_star, h_super_spectrale, h_manhattan
from utils import load, genere_instance



def run_solver(filename, solveur, output):
    """
    Exécute le mode 'solve' sur un fichier unique.
    Écrit la solution dans <output>/chemin_filname.txt.
    """
    print(f"--- Mode Résolution: {filename} ---")

    G, etat = load(filename)
    print(etat)

    if G is None or etat is None:
        print(f"Erreur: Fichier '{filename}' non trouvé ou '0 0' au début.")
        return

    print("Graphe construit. Lancement du solver...")
    start_bfs_time = time.process_time()
    path = solveur(G, (etat['start'], etat['orientation']), etat['goal'])
    end_bfs_time = time.process_time()
    bfs_time = end_bfs_time - start_bfs_time

    if path:
        print("Chemin minimum trouvé:")
        if len(path) > 20:
            print(f"[Chemin de {len(path)-1} actions. Début et fin ci-dessous]")
            print(path[0])
            print("[...]")
            print(path[-1])
        else:
            print(path)
            print(f"\nLongueur du chemin (nombre d'actions): {len(path)-1}")

        output_path = os.path.join(output, f"chemin_{os.path.basename(filename)}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for etape in path:
                f.write(f"{etape}\n")

        print(f"-> Succès : Le chemin a été sauvegardé dans '{output_path}'.")

    else:
        print("--- Aucun chemin trouvé. ---")

    print(f"\nTemps d'exécution : {bfs_time:.6f} secondes")



def worker_task(args):
    """
    Fonction exécutée par un processus du pool.
    Args est un tuple : (N, M, obs_count, solveurs)
    """
    N, M, obs_count, solveurs = args
    results = {}

    try:
        inst = genere_instance(M, N, obs_count)

        # 1. On prépare les arguments "Communs" (Positionnels)
        G = inst['graph']
        start = (inst['start'], inst['orientation'])
        goal = inst['goal']

        # 2. On prépare les arguments "Spéciaux" (Keyword Arguments)
        # On met TOUT ici. Ceux qui en ont besoin les prendront.
        # Les autres les ignoreront grâce au **kwargs.
        context_data = {
            'psi': inst['psi'],
            'eigenvalue': inst['eigenvalue'],
            'h': h_super_spectrale,
            'N_cols': N - 1
        }

        # 3. La boucle propre et universelle
        for solver_func in solveurs:
            solver_name = solver_func.__name__
            start_time = time.process_time()

            solver_func(G, start, goal, **context_data)

            end_time = time.process_time()
            results[solver_name] = end_time - start_time

        return (N, M, obs_count), results, None # Pas d'erreur

    except Exception as e:
        # On capture l'erreur pour ne pas crasher le pool entier
        return (N, M, obs_count), None, e


def run_benchmark(sizes_n, sizes_m, obstacles_list, mode, num_instances, solveurs, output, num_workers, chunk_size=1):
    """
    Exécute le benchmark en parallèle via multiprocessing.Pool.
    """
    print(f"--- Mode Benchmark (Multiprocessing Pool: {num_workers} workers) ---")
    mode = str(mode)

    # 1. Créer les combinaisons de paramètres
    if mode == "u":
        combinations = list(product(sizes_n, sizes_m, obstacles_list))
        print("Mode combinatoire (produit cartésien).")
    elif mode == "v":
        if not (len(sizes_n) == len(sizes_m) == len(obstacles_list)):
            raise ValueError("Mode 'v' requiert que les listes aient la même longueur.")
        combinations = list(zip(sizes_n, sizes_m, obstacles_list))
        print("Mode linéaire (par triplets).")
    else:
        raise ValueError(f"Mode inconnu: {mode}. Utiliser 'u' ou 'v'.")

    # 2. Préparer la liste de toutes les tâches à effectuer
    # Chaque tâche est un tuple d'arguments pour worker_task
    tasks = []
    for N, M, obs_count in combinations:
        for _ in range(num_instances):
            tasks.append((N, M, obs_count, solveurs))

    total_tasks = len(tasks)
    print(f"Préparation de {total_tasks} simulations...")

    results_raw = defaultdict(list)

    # 3. Lancement du Pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_iterator = pool.imap_unordered(worker_task, tasks, chunksize=chunk_size)

        print(f"Exécution en cours...")

        for i, (params, task_results, error) in enumerate(results_iterator, 1):
            N, M, obs_count = params

            if error:
                print(f" [Erreur] Paramètres ({N}, {M}, {obs_count}): {error}")
            else:
                # Agrégation des résultats
                for s_name, duration in task_results.items():
                    key = (s_name, N, M, obs_count)
                    results_raw[key].append(duration)

            # Affichage de progression simple
            if i % max(1, total_tasks // 10) == 0 or i == total_tasks:
                print(f"  Progression : {i}/{total_tasks} ({(i/total_tasks)*100:.0f}%)")

    print("\nBenchmark terminé.")

    # 4. Analyser et sauvegarder
    output_dir = os.path.join(output, "bench")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "resultats_benchmark.csv")

    # Appel des fonctions d'analyse existantes
    results_med = analyser_et_sauvegarder(results_raw, csv_path)

    # Tracer la courbe
    solver_names = [s.__name__ for s in solveurs]
    tracer_courbe(results_med, solver_names, sizes_n, sizes_m, obstacles_list, output_dir, mode)





def analyser_et_sauvegarder(results_raw, path_fichier_res):
    """
    Calcule statistiques (médiane, moyenne, std, count) pour chaque clé
    results_raw: dict {(solver_name, N, M, O): [t1, t2, ...]}
    Écrit un CSV dans path_fichier_res et retourne results_med = {(solver_name, N, M, O): median_time}
    """
    print(f"Analyse des résultats -> écriture CSV: {path_fichier_res}")

    results_med = {}
    stats = {}

    for key, times in results_raw.items():
        arr = np.array(times, dtype=float)
        median = float(np.median(arr))
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        count = int(arr.size)
        stats[key] = (median, mean, std, count)
        results_med[key] = median

    with open(path_fichier_res, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["solver", "N", "M", "Obstacles", "median_s", "mean_s", "std_s", "count"])
        for key in sorted(stats.keys()):
            solver, N, M, O = key
            median, mean, std, count = stats[key]
            writer.writerow([solver, N, M, O, f"{median:.6f}", f"{mean:.6f}", f"{std:.6f}", count])

    return results_med


def tracer_courbe(results_med, solver_names, sizes_n, sizes_m, obstacles_list, plot_dir, mode="u"):
    """
    Trace la courbe de performance.
    Ne trace que si:
      - mode == 'u'  : une seule variable parmi N, M, Obstacles varie (comme avant)
      - mode == 'v'  : on trace les triplets (axe x = index des triplets), avec labels lisibles
    """
    plot_dir = os.path.join(plot_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    mode = str(mode)

    try:
        plt.figure(figsize=(10, 6))

        if mode == "u":
            # Ne tracer que si une seule variable varie
            varying_params = []
            if len(sizes_n) > 1: varying_params.append('N')
            if len(sizes_m) > 1: varying_params.append('M')
            if len(obstacles_list) > 1: varying_params.append('Obstacles')

            if len(varying_params) != 1:
                print(f"\nTraçage ignoré : {len(varying_params)} variables changent.")
                print("Un graphe 2D ne peut être tracé que si une seule dimension varie (mode 'u').")
                plt.close()
                return

            variable_name = varying_params[0]
            x_values = []
            xlabel = ""
            title = ""

            if variable_name == 'N':
                x_values = sorted(sizes_n)
                const_M = sizes_m[0]
                const_O = obstacles_list[0]
                xlabel = "Taille N"
                title = f"Temps Solveur vs Taille N (M={const_M}, Obstacles={const_O})"
                for solver_name in solver_names:
                    y_values = [results_med.get((solver_name, n, const_M, const_O), float('nan')) for n in x_values]
                    plt.plot(x_values, y_values, marker='o', linestyle='-', label=solver_name)

            elif variable_name == 'M':
                x_values = sorted(sizes_m)
                const_N = sizes_n[0]
                const_O = obstacles_list[0]
                xlabel = "Taille M"
                title = f"Temps Solveur vs Taille M (N={const_N}, Obstacles={const_O})"
                for solver_name in solver_names:
                    y_values = [results_med.get((solver_name, const_N, m, const_O), float('nan')) for m in x_values]
                    plt.plot(x_values, y_values, marker='o', linestyle='-', label=solver_name)

            elif variable_name == 'Obstacles':
                x_values = sorted(obstacles_list)
                const_N = sizes_n[0]
                const_M = sizes_m[0]
                xlabel = "Nombre d'Obstacles"
                title = f"Temps Solveur vs Obstacles (Grille {const_N}x{const_M})"
                for solver_name in solver_names:
                    y_values = [results_med.get((solver_name, const_N, const_M, o), float('nan')) for o in x_values]
                    plt.plot(x_values, y_values, marker='o', linestyle='-', label=solver_name)

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("Temps médian (s)")
            plt.grid(True)
            plt.xticks(x_values)
            plt.legend()
            out = os.path.join(plot_dir, f"{variable_name}.png")
            plt.savefig(out)
            plt.close()
            print(f"Courbe de performance sauvegardée : {out}")

        elif mode == "v":
            # mode linéaire: traçage des triplets on met l'index sur l'axe x, avec étiquettes
            if not (len(sizes_n) == len(sizes_m) == len(obstacles_list)):
                print("Traçage mode 'v' ignoré : tailles incohérentes.")
                plt.close()
                return

            x_indices = list(range(len(sizes_n)))
            # créer des labels lisibles pour chaque triplet
            x_labels = [f"N={n},M={m},O={o}" for n, m, o in zip(sizes_n, sizes_m, obstacles_list)]
            title = "Temps Solveur vs Triplets (mode 'v')"
            xlabel = "Instances (N, M, Obstacles)"

            for solver_name in solver_names:
                y_values = []
                for n, m, o in zip(sizes_n, sizes_m, obstacles_list):
                    y = results_med.get((solver_name, n, m, o), float('nan'))
                    y_values.append(y)
                plt.plot(x_indices, y_values, marker='o', linestyle='-', label=solver_name)

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("Temps médian (s)")
            plt.grid(True)
            plt.xticks(x_indices, x_labels, rotation=45, ha='right')
            plt.legend()
            out = os.path.join(plot_dir, "triplets_v.png")
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
            print(f"Courbe (mode 'v') sauvegardée : {out}")

        else:
            print(f"Mode de tracé inconnu: {mode}")
            plt.close()

    except Exception as e:
        print(f"\nErreur lors du traçage de la courbe: {e}")
        try:
            plt.close()
        except:
            pass



SOLVER_MAP = {
    'Bfs_optimise': Bfs_optimise,
    'A_star': A_star,
}

HEURISTIQUE_MAP = {
    'h_manhattan': h_manhattan,
    'h_super_spectrale': h_super_spectrale,
}


def main():
    parser = argparse.ArgumentParser(
        description="Solveur et benchmark pour le problème du robot sur grille.",
        epilog="Exemples d'utilisation: \n"
               "  python votre_script.py solve input.txt --solveur A_star\n"
               "  python votre_script.py bench -n 30 -m 30 -o 50 100 -i 10 --solveurs Bfs A_star"
    )
    # Valide les noms de solveurs disponibles
    available_solvers = list(SOLVER_MAP.keys()) + ['all']

    subparsers = parser.add_subparsers(dest="mode", required=True, 
                                     help="Mode d'exécution")

    # 1. Sous-parser pour le mode 'solve'
    solve_parser = subparsers.add_parser('solve', 
                                         help='Résoudre un problème unique depuis un fichier.')
    solve_parser.add_argument('filename', type=str, 
                              help='Nom du fichier d\'entrée (ex: input.txt)')
    solve_parser.add_argument('-o', '--output', type=str, 
                              default="output",
                              help=f'Dossier de sortie du programme')
    solve_parser.add_argument('-s', '--solveur', type=str, 
                              default=available_solvers[0],
                              choices=available_solvers,
                              help=f'Solveur à utiliser (défaut: {available_solvers[0]})')


    # 2. Sous-parser pour le mode 'bench'
    bench_parser = subparsers.add_parser('bench', 
                                         help='Lancer le benchmark sur des combinaisons de N, M, et Obstacles.')
    bench_parser.add_argument('-o', '--output', type=str, 
                              default="output",
                              help=f'Dossier de sortie du programme')
    bench_parser.add_argument('-n', '--sizes-n', nargs='+', type=int, 
                              help='Liste des tailles N (ex: -n 10 20 30)')
    bench_parser.add_argument('-m', '--sizes-m', nargs='+', type=int, 
                              help='Liste des tailles M (ex: -m 10 20 30)')
    bench_parser.add_argument('-obs', '--obstacles', nargs='+', type=int, 
                              help='Liste des nombres d\'obstacles (ex: -o 50 100 150)')
    bench_parser.add_argument('-t', '--test_type', type=str, default="u",
        choices=["u", "v"],
        help='Mode de test combinatoire (u) ou linéaire (v)')
    bench_parser.add_argument('-i', '--instances', type=int, default=10,
                              help='Nombre d\'instances par combinaison (défaut: 10)')
    bench_parser.add_argument('-s', '--solveurs', nargs='+', type=str, 
                              required=True,
                              dest="solveurs",
                              choices=available_solvers,
                              help='Un ou plusieurs solveur(s) à tester (ex: --solveurs A_star Bfs)')
    bench_parser.add_argument('-w', '--workers', type=int,
                              default=multiprocessing.cpu_count()-1)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.mode == 'solve':
        print("vv", args.output)
        # Convertir le nom (str) en fonction (callback)
        solver_function = SOLVER_MAP[args.solveur]
        # Passer la fonction à run_solver
        run_solver(args.filename, solver_function, args.output)

    elif args.mode == 'bench':
        solver_functions = [] # Liste finale des fonctions

        # Si l'utilisateur a tapé "all"
        if 'all' in args.solveurs:
            solver_functions = list(SOLVER_MAP.values())
            print(f"Test de TOUS les solveurs : {[s.__name__ for s in solver_functions]}")
        else:
            # Sinon, on valide et on mappe manuellement
            try:
                solver_functions = [SOLVER_MAP[name] for name in args.solveurs]
            except KeyError as e:
                # Gérer une erreur si le nom n'existe pas
                print(f"Erreur: Le solveur '{e.args[0]}' n'est pas reconnu.")
                print(f"Solveurs disponibles: {list(SOLVER_MAP.keys())}")
                sys.exit(1)

        run_benchmark(args.sizes_n, 
                      args.sizes_m, 
                      args.obstacles,
                      args.test_type,
                      args.instances, 
                      solveurs=solver_functions,
                      output=args.output,
                      num_workers=args.workers)


if __name__ == "__main__":
    main()
