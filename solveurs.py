from collections import deque
import heapq, math
import numpy as np



orientation_map = {'nord': 0, 'est': 1, 'sud': 2, 'ouest': 3}

# delta[orientation] -> (changement_ligne, changement_colonne)
# Défini comment on bouge selon l'orientation (les index correspondent à ceux donnés par la map orientation_map)
delta_move = [
    (-1, 0), # 0: nord (ligne -1)
    (0, 1),  # 1: est  (col +1)
    (1, 0),  # 2: sud  (ligne +1)
    (0, -1)  # 3: ouest (col -1)
]

# Nombre de pas en avant autorisés en un coup
MAXSTEP = 3




def h_manhattan(args):
    r = args[0];  c = args[1]
    goal_r = args[2];  goal_c = args[3]

    # Calcul Manhattan
    man = abs(r - goal_r) + abs(c - goal_c)

    # Division par MAXSTEP pour l'admissibilité (si on avance de 3 cases max)
    return math.ceil(man / MAXSTEP)


def h_super_spectrale(args):
    """
    Heuristique Hybride : Physique + Spectrale (Optimisation Lookup Table).

    Cette heuristique combine :
    1. La distance réelle (Manhattan) pour la progression locale.
    2. La distance de diffusion (Spectrale) pour la topologie globale (éviter les culs-de-sac).

    Args:
        args (tuple): (pos, goal, spec_map, N_cols, dist_start, beta)
            - pos (tuple): Position actuelle (r, c).
            - spec_map (np.array): Tableau 1D des distances spectrales pré-calculées vers le but.
            - N_cols (int): Largeur du graphe (pour convertir r,c en index 1D).
            - beta (float): Le coefficient d'importance de la distance spectrale (C / lambda_2).

    Returns:
        float: Le coût estimé pondéré.
    """
    # Déballage des arguments nécessaires
    r, c, _, _, spec_map, N_cols, beta = args

    # 1. Calcul de la base physique (Manhattan)
    val_manhattan = h_manhattan(args)

    # 2. "Kill Switch" (Sécurité de proximité)
    # Si l'estimation est < 3 pas, on est tout près.
    # On coupe le spectre pour éviter le micro-bruit numérique et assurer l'atterrissage précis.
    if val_manhattan < 5:
        return val_manhattan

    # 3. Composante Spectrale (Si activée et disponible)
    if spec_map is not None and beta > 0 and N_cols is not None:
        # Conversion index 2D -> 1D
        idx = r * N_cols + c

        # Lecture directe dans la table (O(1))
        if 0 <= idx < len(spec_map):
            d_spec = spec_map[idx]

            # Formule Finale : Physique + (Importance_Difficulté * Distance_Topologique)
            return val_manhattan + (beta * d_spec)

    # Si pas de spectre, on retourne juste Manhattan
    return val_manhattan





def reconstruct_path(came_from, goal_state):
    """
    Parcourt "l'arbre" came_from en arrière pour trouver le chemin.
    """
    path = []
    current = goal_state

    # Remonte le fil
    while current in came_from:
        # Récupère le "parent" et l'action qui a mené à 'current'
        prev_state, action = came_from[current]

        pos = current[0] # La position (r, c) de cet état

        path.append((action, pos))

        current = prev_state # Passe à l'état précédent

        # Le 'None' est le marqueur de début (la racine de l'arbre)
        if current is None:
            break

    path.reverse() # Le chemin a été construit de l'arrivée -> départ
    return path





def Bfs_optimise(G, start, goal, **kwargs):
    """
    Effectue un BFS:
    - Les seules actions sont:
        1. Avancer (1..MAXSTEP)
        2. Tourner (G/D) + Avancer (1..MAXSTEP)
    """
    (start_pos, start_orientation_str) = start

    start_orientation_int = orientation_map[start_orientation_str]
    start_state = (start_pos, start_orientation_int)  # ((r,c), orientation)

    # Initialisation de la file et de la structure de suivi
    queue = deque([start_state])
    came_from = {start_state: (None, 'Start')}

    while queue:
        current_state = queue.popleft()
        (pos, orientation) = current_state

        # Vérification de l'objectif
        if pos == goal:
            return reconstruct_path(came_from, current_state)

        # Action 1 : Avancer (1..MAXSTEP) dans la même direction
        (dr, dc) = delta_move[orientation]
        old_pos = pos
        for step in range(1, MAXSTEP + 1):
            next_pos = (old_pos[0] + dr, old_pos[1] + dc)

            if next_pos in G.get(old_pos, []):
                new_state = (next_pos, orientation)

                if new_state not in came_from:
                    came_from[new_state] = (current_state, f'Avancer({step})')
                    queue.append(new_state)

                    if next_pos == goal:
                        return reconstruct_path(came_from, new_state)

                old_pos = next_pos
            else:
                break

        # Action 2 : Tourner (G/D)
        for new_orientation, action_turn in [
            ((orientation - 1) % 4, 'Tourner Gauche'),
            ((orientation + 1) % 4, 'Tourner Droite')
        ]:
            # 1. On définit le nouvelle état
            turned_state = (pos, new_orientation)

            # 2. On enregistre ce virage dans l'historique s'il n'est pas déjà connu
            if turned_state not in came_from:
                came_from[turned_state] = (current_state, action_turn)
                queue.append(turned_state)

    # Aucun chemin trouvé
    return None




def A_star(G, start, goal, eigenvalue=None, psi=None, h=None, N_cols=None, **kwargs):
    """
    Recherche A*.
    Le robot peut :
        - Avancer tout droit jusqu'à MAXSTEP cases (coût = +1)
        - Tourner à gauche/droite puis avancer (rotation incluse, coût = +2)

    G : Graphe des intersections {sommet: [voisins]}
    start : tuple ( (ligne, col), orientation_str )
    goal : tuple (ligne, col)
    """
    # On définit l'agressivité de l'aide spectrale.
    beta_contextuel = 0.0

    if eigenvalue is not None and len(eigenvalue) > 0:
        # HYPOTHÈSE : eigenvalue a déjà été filtrée dans 'build_structures_normalized' > 0
        # Donc eigenvalue[0] est la première harmonique significative (lambda_2 ou lambda_3...)
        lambda_effective = eigenvalue[0]

        # Paramètres
        C = 0.1             # Sensibilité
        BETA_MAX = 5.0      # Plafond pour éviter de casser l'admissibilité locale

        # Formule : Plus le graphe est déconnecté (lambda petit), plus beta est grand.
        beta_contextuel = min(C / lambda_effective, BETA_MAX)

    # Au lieu de calculer des normes dans la boucle (lent), on pré-calcule
    # la distance spectrale de chaque case vers le but.
    spec_map = None
    if psi is not None and N_cols is not None:
        idx_goal = goal[0] * N_cols + goal[1]
        
        # Vérification bornes
        if 0 <= idx_goal < len(psi):
            goal_vec = psi[idx_goal]

            # Calcul Vectoriel Numpy : || psi_i - psi_goal || pour tout i
            # spec_map est un array 1D. spec_map[idx] = distance spectrale.
            spec_map = np.linalg.norm(psi - goal_vec, axis=1)


    (start_pos, start_orientation_str) = start

    start_orientation_int = orientation_map[start_orientation_str]
    start_state = (start_pos, start_orientation_int)

    h_push = heapq.heappush
    h_pop = heapq.heappop
    (goal_i, goal_j) = goal

    g_start = 0
    f_start = g_start + h((*start_pos, goal_i, goal_j, spec_map, N_cols, beta_contextuel))

    priority_queue = [(f_start, g_start, start_state)]
    cost = {start_state: g_start}
    came_from = {start_state: (None, 'Start')}

    while priority_queue:
        _, g_cost, current_state = h_pop(priority_queue)
        (pos, orientation) = current_state

        if g_cost > cost.get(current_state, float('inf')):
            continue

        if pos == goal:
            return reconstruct_path(came_from, current_state)

        # Cas 1 : tourner (G/D) puis avancer (Coût = g + 2)
        new_g_cost_turn = g_cost + 2

        for new_orientation, action_name in [
            ((orientation - 1) % 4, 'Tourner Gauche'),
            ((orientation + 1) % 4, 'Tourner Droite')
        ]:
            # 1. Calcul de l'état pivot (Position actuelle, Nouvelle orientation)
            pivot_state = (pos, new_orientation)

            # Si on a déjà atteint ce pivot avec un coût inférieur ou égal, 
            # Alors ce virage est inutile. On arrête
            cost_at_pivot = g_cost + 1 

            if cost_at_pivot >= cost.get(pivot_state, float('inf')):
                continue

            # Si on passe ici, c'est le meilleur chemin vers ce pivot
            # On écrase l'historique car on améliore le coût
            cost[pivot_state] = cost_at_pivot
            came_from[pivot_state] = (current_state, action_name)

            # 2. Boucle pour avancer (1 à MAXSTEP)
            (dr, dc) = delta_move[new_orientation]
            old_pos = pos

            for step in range(1, MAXSTEP + 1):
                pos_avant = (old_pos[0] + dr, old_pos[1] + dc)

                if pos_avant not in G.get(old_pos, []):
                    break

                new_state = (pos_avant, new_orientation)

                h_val = h((*pos_avant, goal_i, goal_j, spec_map, N_cols, beta_contextuel))
                new_f = new_g_cost_turn + h_val

                # Si le nouvel état n'a jamais été visité ou qu'il a un coup inférieur
                if new_state not in cost or new_g_cost_turn < cost[new_state]:
                    cost[new_state] = new_g_cost_turn
                    h_push(priority_queue, (new_f, new_g_cost_turn, new_state))

                    # On relie : Pivot -> Nouvelle Position
                    came_from[new_state] = (pivot_state, f"Avancer({step})")

                old_pos = pos_avant


        # Cas 2 : avancer tout droit (Coût = g + 1)
        new_g_cost_move = g_cost + 1
        (dr, dc) = delta_move[orientation]
        old_pos = pos

        for step in range(1, MAXSTEP + 1):
            pos_avant = (old_pos[0] + dr, old_pos[1] + dc)

            if pos_avant not in G.get(old_pos, []):
                break

            new_state = (pos_avant, orientation)
            h_val = h((*pos_avant, goal_i, goal_j, spec_map, N_cols, beta_contextuel))
            new_f = new_g_cost_move + h_val

            if new_state not in cost or new_g_cost_move < cost[new_state]:
                cost[new_state] = new_g_cost_move
                h_push(priority_queue, (new_f, new_g_cost_move, new_state))

                came_from[new_state] = (current_state, f"Avancer({step})")

            old_pos = pos_avant

    return None
