import numpy as np
from collections import defaultdict
import random

from solution import orientation_map, A_star



random.seed(42)


def load(file):
    with open(file, 'r', encoding='utf-8') as f:

        # 1. Lire M (lignes) et N (colonnes)
        ligne_dim = f.readline().strip()

        # Gère le cas où le fichier est vide ou commence par "0 0"
        if not ligne_dim or ligne_dim == '0 0':
            print("Fichier vide ou terminé.")
            return None # On ne peut rien construire

        M_str, N_str = ligne_dim.split()
        M = int(M_str)
        N = int(N_str)

        # 2. Initialiser la grille (M lignes, N colonnes)
        systeme = np.empty((M, N), dtype=int)

        # 3. Lire les M lignes de la grille
        for i in range(M):
            ligne = f.readline().strip().split()
            systeme[i] = ligne

        # 4. Lire la ligne d'état
        state_parts = f.readline().strip().split()

        etat = {
            'start': (int(state_parts[0]), int(state_parts[1])),
            'goal': (int(state_parts[2]), int(state_parts[3])),
            'orientation': state_parts[4]
        }

    return build_graphe(systeme.astype(bool)), etat


def build_graphe(systeme_padded):
    """
    Construire un graphe d'intersections en ne conservant que les noeuds valides
    systeme : np.array booléen (M x N), True = obstacle, False = libre
    Retour : dict { (r,c) : {voisins} }
    """
    M, N = systeme_padded.shape
    graph = defaultdict(set)
    node_valid = set()

    # 1. Calculer les noeuds valides (intersection entourée de 4 cases libres)
    for r in range(1, M):
        for c in range(1, N):
            if not (systeme_padded[r-1, c-1] or systeme_padded[r-1, c] or systeme_padded[r, c-1] or systeme_padded[r, c]):
                node_valid.add((r, c))

    # 2. Construire les arcs uniquement pour les noeuds valides
    for r, c in node_valid:
            # 2-1. Vers la droite
            # Pour aller de (r,c) à (r, c+1), il faut que l'arrivée soit valide
            if (r, c + 1) in node_valid:
                graph[(r, c)].add((r, c + 1))
                graph[(r, c + 1)].add((r, c))

            # 2-2. Vers le bas
            if (r + 1, c) in node_valid:
                graph[(r, c)].add((r + 1, c))
                graph[(r + 1, c)].add((r, c))

    return graph





def verifie_grille(grid, P):
    """
    Vérifie si une grille binaire satisfait les contraintes du problème de PL et affiche les erreurs.

    Paramètres
    ----------
    grid : ndarray, shape (M, N)
        Grille contenant 0/1 (0 = vide, 1 = obstacle)
    P : int
        Nombre total d'obstacles attendu

    Retour
    ------
    bool
        True si la grille respecte toutes les contraintes, False sinon.
    """
    M, N = grid.shape

    # 1. Vérification du nombre total d'obstacles
    total = grid.sum()
    if total != P:
        print(f"Erreur : sum(grid) = {total} != P = {P}")
        return False

    # 2. Maximum par ligne
    ligne_max = grid.sum(axis=1)
    if np.any(ligne_max > 2 * P / M):
        for i, val in enumerate(ligne_max):
            if val > 2 * P / M:
                print(f"Erreur : ligne {i} dépasse 2P/M avec {val} obstacles")
        return False

    # 3. Maximum par colonne
    col_max = grid.sum(axis=0)
    if np.any(col_max > 2 * P / N):
        for j, val in enumerate(col_max):
            if val > 2 * P / N:
                print(f"Erreur : colonne {j} dépasse 2P/N avec {val} obstacles")
        return False

    # 4. Interdiction motif 101 par ligne
    lignes_101 = np.argwhere((grid[:, :-2] == 1) & (grid[:, 1:-1] == 0) & (grid[:, 2:] == 1))
    if len(lignes_101) > 0:
        for i, j in lignes_101:
            print(f"Erreur : motif 101 trouvé sur la ligne {i}, colonnes {j}-{j+2}")
        return False

    # 5. Interdiction motif 101 par colonne
    colonnes_101 = np.argwhere((grid[:-2, :] == 1) & (grid[1:-1, :] == 0) & (grid[2:, :] == 1))
    if len(colonnes_101) > 0:
        for i, j in colonnes_101:
            print(f"Erreur : motif 101 trouvé sur la colonne {j}, lignes {i}-{i+2}")
        return False

    # Si toutes les vérifications passent
    print("Grille valide")
    return True




def genere_instance(M, N, nb_obstacles, min_path_len=None, max_tries=5, verbose=False):
    """
    Génère une instance non triviale et résoluble avec suivi optionnel (verbose)
    """
    if nb_obstacles >= M * N - 2:
        raise ValueError("nb_obstacles trop grand : doit laisser au moins 2 cases libres")

    if min_path_len is None:
        min_path_len = max(M, N) // 8

    def pick_in_topleft(grid):
        """
        Choisit un point libre strictement dans le coin haut-gauche.
        Logique : 5% de la taille ou max 10 cases (pour les grilles géantes).
        """
        # On prend le minimum entre 5% de la taille et 10 cases.
        # max(1, ...) assure qu'on a au moins 1 ligne/colonne sur les toutes petites grilles.
        limit_r = max(1, min(M // 20, 10)) 
        limit_c = max(1, min(N // 20, 10))
        
        rows = range(0, limit_r)
        cols = range(0, limit_c)
        
        choices = [(r,c) for r in rows for c in cols if not grid[r,c]]
        return random.choice(choices) if choices else None

    def pick_in_bottomright(grid):
        """
        Choisit un point libre strictement dans le coin bas-droit.
        """
        limit_r = max(1, min(M // 20, 10))
        limit_c = max(1, min(N // 20, 10))

        # On part de la fin (M) et on remonte de 'limit_r'
        rows = range(M - limit_r, M)
        cols = range(N - limit_c, N)
        
        choices = [(r,c) for r in rows for c in cols if not grid[r,c]]
        return random.choice(choices) if choices else None


    if verbose:
        print(f"\nDébut génération instance : M={M}, N={N}, nb_obstacles={nb_obstacles}, min_path_len={min_path_len}")

    tries = 0
    while tries < max_tries:
        tries += 1
        if verbose:
            print(f"\nEssai {tries}...")

        # 1) Création aléatoire des obstacles
        grid = np.zeros((M, N), dtype=bool)
        if nb_obstacles > 0:
            idx = np.random.choice(M * N, nb_obstacles, replace=False)
            rows = idx // N
            cols = idx % N
            grid[rows, cols] = True

        # 2) Choix de start/goal
        start_pos = pick_in_topleft(grid)
        goal_pos = pick_in_bottomright(grid)

        if start_pos is None or goal_pos is None:
            if verbose:
                print("Échec : aucune case libre disponible dans la zone start ou goal.")
            continue

        # 3) Construire le graphe et vérifier start/goal
        graph = build_graphe(grid)
        if start_pos not in graph or goal_pos not in graph:
            if verbose:
                print("Échec : start ou goal non présent dans le graphe (case bloquée).")
            continue

        # 4) Orientation aléatoire
        orientation = random.choice(list(orientation_map.keys()))

        # 5) Vérifier le chemin via A*
        path = A_star(graph, (start_pos, orientation), goal_pos)
        if path is None:
            if verbose:
                print("Échec : aucun chemin trouvé entre start et goal.")
            continue

        # 6) Vérifier la longueur minimale
        if len(path) < min_path_len:
            if verbose:
                print(f"Échec : chemin trop court ({len(path)} < {min_path_len}).")
            continue

        if verbose:
            print(f"Succès : instance générée en {tries} essais. Longueur chemin = {len(path)}")
        return {
            'M': M,
            'N': N,
            'grid': grid,
            'graph': graph,
            'start': start_pos,
            'goal': goal_pos,
            'orientation': orientation,
            'shortest_length': len(path),
            'tries': tries
        }

    # échec final
    raise RuntimeError(
        f"Impossible de générer une instance valide après {max_tries} essais la grille est trop dense !\n"
        f"Paramètres : M={M}, N={N}, nb_obstacles={nb_obstacles}, min_path_len={min_path_len}"
    )





def get_eigenvector(G):
    """
    Docstring for get_eigenvector
    
    :param G: Description
    """
