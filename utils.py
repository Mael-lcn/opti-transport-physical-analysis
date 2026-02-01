import numpy as np
from collections import defaultdict
import random
from scipy import sparse
from scipy.sparse.linalg import eigsh
import math

from solveurs import orientation_map, A_star, h_manhattan



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

    return build_structures_normalized(systeme.astype(bool)), etat


def build_structures_normalized(systeme, max_k=8, threshold=0.2):
    """
    Construit le graphe de navigation et l'espace spectral sans bordures extérieures.

    Le graphe est basé sur les intersections internes de la grille. Pour une grille
    de M x N cases, il existe (M-1) x (N-1) intersections.

    Args:
        systeme (np.ndarray): Matrice booléenne (M, N) où True = Obstacle.
        max_k (int): Nombre maximum de vecteurs propres à extraire.
        threshold (float): Seuil de coupure pour les valeurs propres significatives.

    Returns:
        tuple: (L_sym, graph_dist1, spectral_vals, spectral_phi)
            - L_sym: Laplacien symétrique normalisé (sparse).
            - graph_dist1: Dictionnaire d'adjacence {(r,c): {voisins}}.
            - spectral_vals: Liste des lambda_i filtrés.
            - spectral_phi: Matrice des vecteurs propres (N_intersections x K).
    """
    M, N = systeme.shape
    # Nombre d'intersections internes (noeuds du graphe)
    n_rows, n_cols = M - 1, N - 1
    num_nodes = n_rows * n_cols

    # --- 1. DÉTECTION DES INTERSECTIONS VALIDES ---
    # Une intersection est valide si les 4 cases adjacentes sont libres (False).
    m1, m2 = systeme[:-1, :-1], systeme[:-1, 1:]
    m3, m4 = systeme[1:, :-1], systeme[1:, 1:]
    mask = ~(m1 | m2 | m3 | m4)

    # --- 2. GÉNÉRATION DES MASQUES DE CONNECTIVITÉ (SLICING) ---
    # Pas de distance 1
    h_step1 = mask[:, :-1] & mask[:, 1:]
    v_step1 = mask[:-1, :] & mask[1:, :]

    # Sauts de distance 2 et 3 (validation du segment complet)
    h_step2 = h_step1[:, :-1] & h_step1[:, 1:]
    v_step2 = v_step1[:-1, :] & v_step1[1:, :]
    h_step3 = h_step2[:, :-1] & h_step1[:, 2:]
    v_step3 = v_step2[:-1, :] & v_step1[2:, :]


    # --- 3. DICTIONNAIRE DE RECHERCHE (DISTANCE 1 UNIQUEMENT) ---
    graph_dist1 = defaultdict(set)
    for r, c in np.argwhere(h_step1):
        u, v = (r, c), (r, c + 1)
        graph_dist1[u].add(v); graph_dist1[v].add(u)
    for r, c in np.argwhere(v_step1):
        u, v = (r, c), (r + 1, c)
        graph_dist1[u].add(v); graph_dist1[v].add(u)


    # --- 4. CONSTRUCTION DU LAPLACIEN NORMALISÉ (AVEC SAUTS 1-3) ---
    def get_edges(step_mask, dr, dc):
        """Calcule les indices linéaires source/cible pour le rectangle interne."""
        r, c = np.where(step_mask)
        idx_src = r * n_cols + c
        idx_tgt = (r + dr) * n_cols + (c + dc)
        return idx_src, idx_tgt


    srcs, tgts = [], []
    for step, dr, dc in [(h_step1, 0, 1), (h_step2, 0, 2), (h_step3, 0, 3),
                         (v_step1, 1, 0), (v_step2, 2, 0), (v_step3, 3, 0)]:
        s, t = get_edges(step, dr, dc)
        srcs.extend(s); tgts.extend(t)
        srcs.extend(t); tgts.extend(s)


    # Matrice d'adjacence creuse (CSR)
    A = sparse.coo_matrix((np.ones(len(srcs)), (srcs, tgts)), 
                          shape=(num_nodes, num_nodes)).tocsr()
    A.data = np.ones_like(A.data) # Unifie les poids

    # Calcul des degrés et normalisation symétrique
    d = np.array(A.sum(axis=1)).flatten()
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0

    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    L_sym = sparse.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

    # --- 5. ANALYSE ET EXTRACTION SPECTRALE ---
    spectral_vals, spectral_phi = [], None
    phi = None

    try:
        # On demande un peu plus de valeurs propres pour être sûr d'avoir des non-nulles
        k_target = min(max_k, num_nodes - 1) 

        if num_nodes < 1000:
            vals, vects = np.linalg.eigh(L_sym.toarray())
            vals = vals[:k_target]
            vects = vects[:, :k_target]
        else:
            vals, vects = eigsh(L_sym.astype(float), k=k_target, which='SM', tol=1e-2)

        # Tri croissant
        idx_sorted = np.argsort(vals)
        vals, vects = vals[idx_sorted], vects[:, idx_sorted]

        #On enlève tout ce qui est proche de 0
        epsilon = 1e-6
        mask_non_zero = vals > epsilon

        clean_vals = vals[mask_non_zero]
        clean_vects = vects[:, mask_non_zero]

        if len(clean_vals) == 0:
            print("Attention: Graphe totalement déconnecté ou trivial.")
            return L_sym, graph_dist1, [], None

        # On ne garde que les basses fréquences utiles pour la géométrie globale
        mask_threshold = clean_vals < threshold
        spectral_vals = clean_vals[mask_threshold]
        spectral_phi = clean_vects[:, mask_threshold]

        # Sécurité : Si on a tout filtré (trop sévère), on garde au moins les 3 premiers non-nuls
        if len(spectral_vals) < 3 and len(clean_vals) >= 1:
            limit = min(3, len(clean_vals))
            spectral_vals = clean_vals[:limit]
            spectral_phi = clean_vects[:, :limit]

        # Préparation finale des vecteurs pondérés
        phi = prepare_spectral_gps(spectral_phi, spectral_vals)

    except Exception as e:
        print(f"Erreur convergence spectrale : {e}")

    return L_sym, graph_dist1, spectral_vals, phi


def prepare_spectral_gps(spectral_phi, spectral_vals):
    """
    Pré-calcule la matrice GPS pondérée Psi.
    Distance(u,v) = ||Psi[u] - Psi[v]||_2
    """
    # Division chaque colonne de Phi par sqrt(lambda_i)
    return spectral_phi / np.sqrt(spectral_vals)





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
        L, graph, eigenvalue, psi = build_structures_normalized(grid)
        if start_pos not in graph or goal_pos not in graph:
            if verbose:
                print("Échec : start ou goal non présent dans le graphe (case bloquée).")
            continue

        # 4) Orientation aléatoire
        orientation = random.choice(list(orientation_map.keys()))

        # 5) Vérifier le chemin via A*
        path = A_star(graph, (start_pos, orientation), goal_pos, eigenvalue, psi, h_manhattan, N-1)
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
            'eigenvalue': eigenvalue,
            'psi': psi,
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
