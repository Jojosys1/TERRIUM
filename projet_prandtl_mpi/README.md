# Projet 2 (MSN) — Équation intégro-différentielle de Prandtl en parallèle
Fortran 2008 · `mpi_f08` · topologie cartésienne 1D · `MPI_Iallgatherv` avec recouvrement · métrologie `MPI_Wtime`

## Contenu
```
src/precision_mod.f90      précision (dp) et constantes
src/parametres_mod.f90     paramètres du cas, lus sur la ligne de commande (cle=valeur)
src/metrologie_mod.f90     temps par phase, réduction du pire temps (MPI_MAX), médiane
src/decomposition_mod.f90  MPI_Cart_create / MPI_Cart_shift, découpage de l'envergure en blocs
src/aile_mod.f90           corde c(y), incidence utile, surface de référence
src/solveur_mod.f90        produit matrice-vecteur distribué, gradient conjugué, CL et CDi
src/main.f90               mesures chronométrées, bilan, CSV, profil de circulation
Makefile                   all | debug | test | clean
slurm/strong_scaling.sh    passage à l'échelle forte (N fixe)
slurm/weak_scaling.sh      passage à l'échelle faible (N = N0·√P)
analysis/analyse_scaling.py  speedup, efficacité, f d'Amdahl, Karp-Flatt, Gustafson, figures
tests/valider.sh, glauert.py validation automatique
```

## Modèle
Ligne portante de Prandtl, envergure y ∈ [−b/2, b/2], avec g = Γ/V∞ :

    α(y) − α_L0 = 2 Γ(y) / (a0 V∞ c(y))  +  (1/(4π V∞)) · VP ∫ Γ'(η)/(y − η) dη

- N panneaux uniformes, Γ constant par panneau : Γ' devient une somme de **sauts** `dg_k = g_k − g_{k−1}` aux noeuds ; le point de collocation (centre du panneau) n'est jamais sur un noeud, ce qui donne la valeur principale de Cauchy sans singularité (c'est le rôle du « +0,5 » du squelette du sujet).
- Système : `(A g)_i = 2 g_i/(a0 c_i) + 1/(4πh) · Σ_k dg_k / (i − k − ½) = α_i − α_L0`. La matrice est symétrique définie positive : **gradient conjugué**.
- Coefficients : `CL = 2 Σ g_i h / S`, `CDi = 2 Σ g_i α_ind,i h / S` avec `α_ind = rhs − 2g/(a0 c)`.

## Parallélisme
- Rang r possède un bloc contigu de panneaux ; les restes de la division sont répartis sur les premiers rangs.
- Chaque produit A·x : (1) `MPI_Isend/Irecv` du dernier panneau vers le voisin de droite (`MPI_Cart_shift`) ; (2) sauts locaux ; (3) `MPI_Iallgatherv` des sauts ; (4) pendant le transfert, interactions avec les **noeuds locaux** (avec des `MPI_Test` périodiques pour faire progresser l'asynchrone) ; (5) `MPI_Wait` ; (6) interactions avec les noeuds distants.
- Coût par produit : O(N²/P) en calcul, O(N) en communication. Deux `MPI_Allreduce` par itération (produits scalaires).
- `recouvrement=0` remplace l'étape 3-5 par un `MPI_Allgatherv` bloquant : comparaison directe dans l'audit.

## Compilation et exécution
```
make                        # -O3 -march=native -std=f2008 -Wall -Wextra -pedantic (aucun avertissement)
make debug                  # contrôles d'indices et exceptions flottantes
make test                   # validation automatique
mpirun -np 4 ./bin/prandtl n=4096 incidence=5 rep=5 dump=1
```
Paramètres (tous optionnels) : `n` (pair), `envergure`, `croot`, `ctip`, `incidence`, `vrillage` (au saumon, deg), `alpha_l0`, `a0`, `elliptique`, `vinf`, `rho`, `tol`, `itmax`, `rep`, `recouvrement`, `dump`, `sortie`.
`-march=native` : compiler sur une machine du type des noeuds de calcul, sinon retirer l'option du Makefile.

## Métrologie
`MPI_Barrier` → `MPI_Wtime` → résolution → `MPI_Barrier` → `MPI_Wtime`, puis `MPI_Reduce(MPI_MAX)` : **pire temps** des rangs. Une résolution de chauffe non chronométrée précède les `rep` répétitions ; on retient le minimum et la médiane. Le bilan donne aussi, par phase (pire rang) : calcul local, calcul distant, attente (`MPI_Wait`), halo, réductions. Si `attente` reste faible devant `calcul local`, la communication est bien masquée.

## Passage à l'échelle
```
sbatch slurm/strong_scaling.sh          # N=8192, P = 1,2,4,8,16
sbatch slurm/weak_scaling.sh            # N = N0·√P, N0=4096
python3 analysis/analyse_scaling.py data/strong_N8192.csv --mode strong
python3 analysis/analyse_scaling.py data/weak_N0_4096.csv --mode weak
```
- **Faible** : le coût est en N²/P, donc N ∝ √P (et non N ∝ P) garde la charge par coeur constante.
- L'analyse utilise le temps **par itération** : le nombre d'itérations du gradient conjugué croît avec N (≈ √N), ce qui pénaliserait à tort le passage à l'échelle faible.
- Karp-Flatt constant en P ⇒ fraction séquentielle ; croissant en P ⇒ surcoût de communication (réponse au débat « logiciel ou matériel »).
- Dans ce code presque tout est parallèle (seules les E/S de fin de calcul sont séquentielles) : si un décrochage apparaît, l'hypothèse à tester est l'`allgather` (volume N par processus) et les deux réductions par itération, via `t_attente` et la métrique de Karp-Flatt.

## Validation (gfortran 13.3, OpenMPI 4.1.6)
| Contrôle | Résultat |
|---|---|
| Aile elliptique, N = 500 → 4000 | écart de CL à la théorie 0,044 → 0,022 → 0,011 → 0,005 % (ordre 1), e = CL²/(πλCDi) → 1,0002 |
| P = 1, 2, 3, 4, 7 × recouvrement 0/1 (N = 1000) | mêmes CL, CDi et nombre d'itérations (écart < 1e-9) |
| Trapézoïdale vrillée/cambrée, 4 cas vs Glauert (Fourier) | écart ≤ 0,02 % sur CL et CDi |
| `make debug` (`-fcheck=all`, pièges FPE), P = 1, 3, 4 | aucune erreur |

**Limite de ce banc d'essai : un seul cœur disponible, donc aucun speedup n'a pu être mesuré.** Les campagnes de scaling sont à lancer sur le cluster de l'UNSTIM.

## Choix de conception et points à confirmer avec l'enseignant
1. **Forme de l'équation.** Le sujet écrit le terme local `Γ/(2b V∞)` ; j'utilise la forme standard `2Γ/(a0 V∞ c(y))` (avec a0 = 2π : `Γ/(π V∞ c)`), qui coïncide avec celle du sujet pour une corde constante c = 2b/π. À confirmer si la forme du sujet est imposée.
2. **Flèche et dièdre** n'interviennent pas dans la ligne portante classique ; seuls corde, vrillage et incidence de portance nulle comptent. L'application Streamlit (VLM 3D) les prend en compte.
3. **Solveur** : gradient conjugué plutôt qu'un schéma de point fixe, dont la convergence se dégrade quand N augmente.
4. **N pair** (un noeud en y = 0), maillage uniforme : convergence d'ordre 1 en h.
5. **Fortran seulement.** Le sujet parle de binôme « C++17 ou Fortran 2008 » et ailleurs de « bilingue » : à clarifier si la version C++ est exigée.
