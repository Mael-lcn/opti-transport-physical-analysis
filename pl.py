from gurobipy import *
import numpy as np



def proglin(M, N, P):
    """
    Résout un MIP binaire sur une grille MxN :
      - variables x_{i,j} ∈ {0,1} (obstacle=1)
      - contraintes : somme par ligne <= 2*P/M
                      somme par colonne <= 2*P/N
      - interdiction de motifs "101" horizontal et vertical
    Retourne la grille solution (np.ndarray shape (M,N) d'entiers 0/1).

    Paramètres
    ----------
    M, N : int
        dimensions de la grille (M lignes, N colonnes).
    P : int
        Nombre d'obstacle.
    """
    if M <= 0 or N <= 0:
        raise ValueError("M et N doivent être strictement positifs")

    K = M * N
    c = np.random.randint(0, 1001, (M, N))

    m = Model("mogplex")

    # Créer les variables binaires
    x = m.addVars(K, vtype=GRB.BINARY, name="x")

    # Contraintes lignes (somme <= 2*P/M)
    for i in range(M):
        indices = [i*N + j for j in range(N)]
        m.addConstr(quicksum(x[j] for j in indices) <= 2*P/M, name=f"row_{i}")

    # Contraintes colonnes (somme <= 2*P/N)
    for j in range(N):
        indices = [i*N + j for i in range(M)]
        m.addConstr(quicksum(x[i] for i in indices) <= 2*P/N, name=f"col_{j}")

    # Interdictions "101" sur les lignes
    for i in range(M):
        for j in range(N - 2):
            m.addConstr(x[i*N + j] + x[i*N + j + 2] <= 1 + x[i*N + j + 1], name=f"row101_{i}_{j}")

    # Interdictions "101" sur les colonnes
    for j in range(N):
        for i in range(M - 2):
            m.addConstr(x[i*N + j] + x[(i+2)*N + j] <= 1 + x[(i+1)*N + j], name=f"col101_{i}_{j}")

    m.addConstr(quicksum(x[i] for i in range(K)) == P)

    # Objectif : min sum c * x
    obj = quicksum(float(c[i,j]) * x[i*N + j] for i in range(M) for j in range(N))
    m.setObjective(obj, GRB.MINIMIZE)

    # Résolution
    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise Exception(f"Le PL n'a pas été résolu de manière optimale. Statut du solveur : {m.status}")

    # Extraction de la grille 0/1
    system = np.zeros((M, N), dtype=bool)
    for i in range(M):
        for j in range(N):
            system[i, j] = int(x[i*N + j].x)

    return system
