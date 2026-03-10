# Projet Final PHS3903

Le projet consiste à simuler une centaine de molécules d'eau à leur transistion de phase liquide-gaz et gaz-liquide.

<!-- ## Navigation du projet

Les noms seront à changer

`graphs`: Graphiques tracés à partir des résultats et du post-processing
`parameters`: Paramètres 
`post-processing`:
`processing`:
`results`:
`time-evolution`:
`visualisation`: -->


# (WIP) Tâches à réaliser:
### Infrastucture de simulation:
- Définition du vecteur d'état du système
- Définition des conditions initales (Slab?)
- Permet de générer l'état du système dans le temps et le store dans un fichier log.

### Implémentation du modèle d'eau:
- SPC/E
    - Calculs de potentiels/gradients

---
### Est-ce qu'il faudrait pas mettre Thermostat et intégrateur temporel ensemble? À la fin c'est les équations du mouvement qu'on code?
---

### Thermostat:
- Nosé-Hoover

### Intégrateur temporel
- Verlet

### Calculs
- Température
- Entropie
- Pression
- Capacité calorifique
- +Graphiques

### Interface graphique
- Affichage des molécules selon l'état du système

## Semaine 9 (09/03 - 15/03):
### (11/03) RAPPORT INTERIMAIRE


## Semaine 10 (16/03 - 22/03):

## Semaine 11 (23/03 - 29/03):

## Semaine 12 (30/03 - 05/04):

## Semaine 13 (06/04 - 12/04):
### Optimisation
- Réécriture des étapes critiques du code en Julia.
- Utilisation de CuPy ou Cuda.jl
### Comparaison
- Ajout d'autres modèles d'eau (TIP4P/2005)
- Ajout d'autres thermostats

## Semaine 14 (13/04 - 19/04):
### (16/04) CAHIER DE PROJET
### (16/04) EVALUATION PAR LES PAIRS
### (19/04) DIAPOSITIVES PRESENTATION ORALE 
## Semiane 15 (20/04 - 26/04):