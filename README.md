# Terrium — simulation aérodynamique d'une aile de drone, d'un avion ou d'un aile d'un pallier éolien

Application Streamlit de visualisation d'une aile de drone, avec :

- génération paramétrique de la géométrie et du maillage surfacique ;
- visualisation 3D de l'écoulement potentiel par VLM ;
- solveur Fortran 2008 parallèle MPI pour l'équation de ligne portante de Prandtl ;
- comparaison de scénarios, d'incidences et de maillages 1D ;
- affichage des profils de circulation `Gamma(y)` et des gradients de circulation ;
- comparaison avec la théorie elliptique et la solution analytique de Glauert.

Le calcul scientifique du solveur de Prandtl reste réalisé en Fortran. Python
pilote la compilation et l'exécution MPI, récupère les CSV et affiche les
résultats dans l'onglet **Solveur**.

## Structure

```text
.
|-- app_aile_drone_fct.py       # application Streamlit principale
|-- fortran_bridge.py           # compilation, lancement et parsing Fortran
|-- requirements.txt            # dépendances Python
|-- aile_drone.stl              # géométrie téléchargeable
|-- aile_noeuds.csv             # export de nœuds disponible dans l'interface
|-- aile_elements.csv           # export d'éléments disponible dans l'interface
`-- projet_prandtl_mpi/
    |-- src/                    # sources Fortran MPI
    |-- tests/                  # référence Glauert et validations
    |-- analysis/               # analyse optionnelle des campagnes Slurm
    |-- slurm/                  # scripts de scaling pour cluster
    |-- Makefile
    `-- README.md               # documentation technique du solveur
```

Les dossiers `build*`, `bin` et les fichiers CSV produits par une exécution
sont ignorés par Git. Ils sont recréés automatiquement.

## Prérequis

### Application Streamlit

- Python 3.10 ou plus récent ;
- `pip` ;
- Windows, Linux ou macOS.

### Solveur Fortran/MPI

- compilateur Fortran MPI accessible dans le `PATH` :
  - `mpif90` ou `mpifort` ;
- lanceur MPI pour plusieurs processus :
  - `mpiexec` ou `mpirun`.

Sous Windows, une installation MSYS2/MinGW avec `mpif90` et Microsoft MPI
avec `mpiexec` peut être utilisée. Sous Linux, installer généralement
`gfortran`, OpenMPI et leurs paquets de développement.

## Installation Python

Depuis la racine du projet :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sous Linux/macOS :

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Lancer l'application

```powershell
streamlit run app_aile_drone_fct.py
```

L'onglet **Solveur** affiche les outils détectés, les paramètres transmis au
Fortran, les journaux, les métriques MPI, le profil de circulation, les
gradients et les campagnes de comparaison.

## Compilation du solveur Fortran

### Depuis l'application

L'application compile automatiquement les sources Fortran lorsque le bouton
**Compiler et lancer le solveur Fortran** ou le bouton de campagne est utilisé.
La compilation est ignorée si le binaire est plus récent que toutes les
sources.

### Depuis `projet_prandtl_mpi`

Sur Linux ou dans un environnement contenant GNU Make :

```bash
cd projet_prandtl_mpi
make
```

Compilation de mise au point :

```bash
make debug
```

Nettoyage des artefacts générés :

```bash
make clean
```

Sous Windows, si `make` n'est pas installé, l'application utilise directement
`mpif90` et produit `projet_prandtl_mpi/bin/prandtl_streamlit.exe`.

## Exécution directe

Exemple avec quatre processus MPI :

```bash
mpiexec -np 4 ./bin/prandtl n=1024 incidence=5 rep=3 dump=1
```

Avec le binaire construit par l'application sous Windows :

```powershell
mpiexec -np 4 .\bin\prandtl_streamlit.exe n=1024 incidence=5 rep=3 dump=1
```

Paramètres principaux :

| Paramètre | Description |
|---|---|
| `n` | nombre de sections 1D, automatiquement rendu pair |
| `envergure` | envergure totale `b` en mètres |
| `croot`, `ctip` | cordes à la racine et au saumon |
| `incidence` | incidence géométrique en degrés |
| `vrillage` | vrillage au saumon en degrés |
| `alpha_l0` | incidence de portance nulle |
| `a0` | pente de portance 2D |
| `elliptique` | `0` trapézoïdale, `1` elliptique |
| `vinf` | vitesse de référence |
| `rho` | masse volumique |
| `tol`, `itmax` | tolérance et nombre maximal d'itérations |
| `rep` | répétitions chronométrées |
| `recouvrement` | `1` communication non bloquante, `0` bloquante |
| `dump` | `1` pour produire les profils CSV |
| `sortie` | chemin du fichier de résultats |

## Comparaison des scénarios

Dans **Solveur**, la campagne peut faire varier :

- plusieurs angles d'incidence ;
- une géométrie trapézoïdale ou elliptique ;
- plusieurs maillages 1D `N` ;
- plusieurs nombres de processus MPI.

Chaque valeur `N` correspond au nombre de sections de la ligne portante
Fortran, et non au nombre de panneaux de la visualisation VLM. Les combinaisons
déjà exécutées sont conservées dans le cache de session Streamlit.

Les références affichées sont :

- théorie elliptique pour l'aile elliptique sans vrillage ;
- série de Fourier de Glauert pour l'aile trapézoïdale.

Les écarts de `CL` et `CDi` sont indiqués en pourcentage. Les résultats
Fortran et les résultats du modèle VLM 3D ne sont pas mélangés : ils reposent
sur deux modèles numériques différents.

## Validation

Validation Python :

```powershell
python -m py_compile app_aile_drone_fct.py fortran_bridge.py
```

Validation Fortran depuis le dossier du solveur :

```bash
make test
```

Le script de validation contrôle notamment :

- le cas elliptique ;
- l'indépendance des résultats par rapport au nombre de processus MPI ;
- les modes avec et sans recouvrement ;
- la comparaison trapézoïdale avec Glauert.

## Fichiers produits à l'exécution

Lorsque l'export est activé, le Fortran produit :

- `resultats.csv` : métriques numériques et temporelles ;
- `profil_gamma.csv` : `y`, corde, circulation normalisée et incidence induite ;
- `profil_gradient.csv` : sauts de circulation, gradients et rang MPI propriétaire.

Ces fichiers sont temporaires et sont récupérés automatiquement par Python.

## Limites connues

- Le solveur Fortran est un modèle de ligne portante 1D ; il ne représente pas
  directement la flèche, le dièdre, le profil NACA ou le maillage en corde VLM.
- Le modèle Python/VLM est une visualisation 3D distincte.
- Le nombre de processus MPI doit être inférieur ou égal au nombre de sections
  `N`.
- Un maillage `N` impair est automatiquement augmenté à la valeur paire
  suivante.
