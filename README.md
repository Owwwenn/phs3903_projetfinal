# Projet Final PHS3903

Le projet consiste à simuler une centaine de molécules d'eau à leur transistion de phase liquide-gaz et gaz-liquide.

# Setup
Après avoir cloné le repo, créer une venv pour le projet pour éviter les problèmes de dépendances.

### 0- Installer virtualenv

```
pip install virtualenv
```

### 1- Dans le dossier du projet (cd dans la bonne dir si besoin) - Créer la venv
```
python -m venv venv
```
### 2- Activer la venv:

Windows:

```
.\venv\Scripts\activate
```

Linux:

```
source .venv/bin/activate
```

### 3- Installer les dépendances dans la venv

```
pip install -r requirements.txt
```

### 4- Installer le répo comme package dans la venv

```
pip install -e .
```

Si tout fonctionne, il ne devrait pas y avoir de problèmes avec les imports de fonctions depuis les autres fichier.

# Mode d'emploi 
Modifier les paramètres dans `scripts/main.py` pour lancer une simulation. Les autres scripts servent à générer les benchmarks, des tests ou des résultats (graphiques).

Les résultats apparaissent dans `results/`.

L'ensemble des codes réutilisés comme package est dans `src/`.