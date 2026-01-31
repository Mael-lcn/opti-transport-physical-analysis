import os

from solution import A_star, genere_instance
from bench import run_solver

try:
    from pl import proglin
    HAS_Gurobi = True
except Exception:
    print("Impossible d'importer Gurobi, cela n'aura pas d'impact sur la suite.")
    HAS_Gurobi = False



def choix_mode():
    print("\n=== MODE DE FONCTIONNEMENT ===")
    print("1) Charger une instance depuis un fichier")
    print("2) Générer une instance automatiquement avec Gurobi (PL)")

    while True:
        choix = input("Votre choix (1 ou 2) : ").strip()
        if choix in ["1", "2"]:
            return choix
        print("Choix invalide. Entrez 1 ou 2.")


def parametres_grille():
    print("\nEntrer les paramètres de la grille.")

    while True:
        try:
            m = int(input("Nombre de lignes M : "))
            if m <= 3:
                print("M doit être un entier strictement > 3.")
                continue
            break
        except ValueError:
            print("Veuillez entrer un entier valide.")

    while True:
        try:
            n = int(input("Nombre de colonnes N : "))
            if n <= 3:
                print("N doit être un entier strictement > 3.")
                continue
            break
        except ValueError:
            print("Veuillez entrer un entier valide.")

    while True:
        try:
            p = int(input("Nombre d'obstacles P : "))
            if p < 0:
                print("P ne peut pas être négatif.")
                continue
            if p >= (m * n)//2:
                print(f"P ne peut pas dépasser {(m * n)//2}.")
                continue
            break
        except ValueError:
            print("Veuillez entrer un entier valide.")

    return m, n, p


def demander_position(message, m, n, grille):
    """
    Demande à l'utilisateur une position valide sur la grille.
    Vérifie les bornes et la présence d'obstacles.
    """
    print(message)
    while True:
        try:
            x, y = map(int, input("Entrez la position (ligne colonne) : ").split())
        except ValueError:
            print("Entrée invalide. Donnez deux entiers (ex : 3 4).")
            continue

        if not (0 < x < m):
            print(f"Ligne hors bornes : {x}. Doit être entre 1 et {m-1}.")
            continue

        if not (0 < y < n):
            print(f"Colonne hors bornes : {y}. Doit être entre 1 et {n-1}.")
            continue

        if grille[x, y] != 0:
            print(f"La case ({x}, {y}) est un obstacle.")
            continue

        return x, y


def parametres_robots(m, n, grille):
    # Départ
    x_d, y_d = demander_position("Point de départ sous forme ligne colonne (ex : 2 3) :", m, n, grille)
    # Arrivée
    x_a, y_a = demander_position("Point d'arrivée sous forme ligne colonne (ex : 2 3) :", m, n, grille)

    # Orientation
    print("\nChoisir l'orientation initiale parmi : nord, sud, est, ouest")
    o = input("Orientation initiale : ").lower()
    while o not in ['nord', 'sud', 'est', 'ouest']:
        print(f"Orientation '{o}' invalide.")
        o = input("Réessaie (nord/sud/est/ouest) : ").lower()

    return (x_d, y_d), (x_a, y_a), o


def ecrire_fichier_instance(filename, grille, M, N, start, goal, o):
    """
    Écrit une instance dans un fichier compatible avec run_solver / load().
    
    Paramètres :
    - filename : chemin du fichier de sortie
    - grille : matrice M x N (listes de listes ou numpy array) avec 0 = libre, 1 = obstacle
    - start : tuple (ligne, colonne) du départ
    - goal : tuple (ligne, colonne) de l'arrivée
    - orientation : string 'nord', 'sud', 'est', 'ouest'
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        # ligne et colonne
        f.write(f"{M} {N}\n")
        
        # grille
        for ligne in grille:
            ligne_str = " ".join(str(x) for x in ligne)
            f.write(ligne_str + "\n")
        
        # points dep et arrivee, orientation
        x_d, y_d = start
        x_a,y_a = goal
        f.write(f"{x_d} {y_d} {x_a} {y_a} {o}\n")

        # fin
        f.write("0 0\n")

    print(f"Fichier '{filename}' créé avec succès !")



def main() :
    os.makedirs("output", exist_ok=True)
    choix = choix_mode()

    # MODE 1 : Charger un fichier existant
    if choix == "1":
        while True:
            try:
                fichier = input("Chemin du fichier d'instance : ").strip()
                print("\nRésolution de l'instance...")
                run_solver(fichier, A_star, "output")
                return 0  # succès -> on quitte

            except Exception as e:
                print(f"\nUne erreur est survenue pendant la résolution : {e}")

    # MODE 2 : Générer une instance via PL (Gurobi)
    while True:  # boucle principale pour redemander m,n,P si besoin
            m, n, P = parametres_grille()  # saisie des paramètres

            # génération de la grille : si échec on remonte et redemande m,n,P
            try:
                if HAS_Gurobi:
                    print("\nGénération PL de la grille avec Gurobi...")
                    grille = proglin(m, n, P)
                else:
                    print("\nGénération de la grille sans Gurobi...")
                    grille = genere_instance(m, n, P)['grid']
            except Exception as e_gen:
                print(f"\n Erreur lors de la génération de la grille : {e_gen}")
                print(" On va redemander les paramètres m, n et P.")
                continue  # recommence la boucle principale (redemande m,n,P)
            
            grille = grille.astype(int)
            # paramètres robot (saisie) — si l'utilisateur se plante, parametres_robots gère normalement les retries
            start, goal, o = parametres_robots(m, n, grille)

            # boucle dédiée à l'écriture du fichier (on redemande juste le nom de fichier si écriture échoue)
            while True:
                try:
                    filename = input("\nOù voulez-vous enregistrer votre grille ? (ex : input/instance1.txt) : ").strip()
                    if filename == "":
                        filename = "utilisateur.txt"
                    ecrire_fichier_instance(filename, grille, m, n, start, goal, o)
                    break  # fichier écrit avec succès -> sortir de la boucle d'écriture
                except KeyboardInterrupt:
                    print("\nInterrompu par l'utilisateur.")
                    return 1
                except Exception as e_file:
                    print(f"\nImpossible d'écrire le fichier : {e_file}")
                    print("Veuillez ressaisir un nom de fichier valide.")
                    # la boucle d'écriture recommence

            # lancement du solver; si ça plante on revient à la saisie des paramètres (m,n,P)
            try:
                print("\nRésolution avec A*...")
                run_solver(filename, A_star, "output")
                return 0  # tout s'est bien passé -> sortie

            except Exception as e_solver:
                print(f"\nErreur lors de la résolution : {e_solver}")
                print("On va redemander m, n, P et régénérer la grille.")
                continue  # recommence la boucle principale (redemande m,n,P)


if __name__ == "__main__":
    main()
