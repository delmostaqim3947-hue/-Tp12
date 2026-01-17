import numpy as np

# Noms et notes (4 étudiants x 3 épreuves)
noms = np.array(["Alice", "Bob", "Chloé", "David"])
notes = np.array([
    [14.5, 16.0, 13.0], # Alice
    [11.0, 10.5, 12.5], # Bob
    [17.0, 18.5, 16.5], # Chloé
    [9.0, 11.0, 10.0]   # David
])

print("Shape des notes :", notes.shape) # Affiche (4, 3)
# Statistiques par épreuve (Axe 0)
moyennes_epreuves = np.mean(notes, axis=0)

# Statistiques par étudiant (Axe 1)
moyennes_etudiants = np.mean(notes, axis=1)
std_etudiants = np.std(notes, axis=1)
# Meilleur étudiant
idx_top = np.argmax(moyennes_etudiants)
print(f"Top étudiant : {noms[idx_top]} ({moyennes_etudiants[idx_top]:.2f})")

# Épreuve la plus difficile
epreuves = np.array(["E1", "E2", "E3"])
idx_difficile = np.argmin(moyennes_epreuves)
print(f"Épreuve difficile : {epreuves[idx_difficile]}")
# Etudiants ayant une moyenne >= 12
masque_moyenne = moyennes_etudiants >= 12
etudiants_admis = noms[masque_moyenne]

# Etudiants ayant eu >= 15 à l'épreuve E2 (index 1)
masque_e2 = notes[:, 1] >= 15
print("Admis en E2 :", noms[masque_e2])
# 1. Ajout de la colonne Moyenne (reshape nécessaire pour hstack)
moy_col = moyennes_etudiants.reshape(-1, 1)
notes_enrichies = np.hstack([notes, moy_col])

# 2. Ajout de la ligne de statistiques en bas
moy_global = np.mean(moyennes_etudiants)
ligne_moy = np.append(moyennes_epreuves, [moy_global])
tableau_final = np.vstack([notes_enrichies, ligne_moy])

# 3. Préparation des labels
noms_ext = np.append(noms, "Moyennes globales")
print("\n--- RAPPORT FINAL ---")
for nom, ligne in zip(noms_ext, tableau_final):
    valeurs = " | ".join(f"{val:6.2f}" for val in ligne)
    print(f"{nom:>20} : {valeurs}")
