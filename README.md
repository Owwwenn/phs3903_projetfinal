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

### Thermostat:
- Nosé-Hoover
- Définit les matrices des équations du mouvement que Verlet utilise (à vérifier)

### Intégrateur temporel
- Verlet
- Résolution des équations du mouvement à un pas de temps (à vérifier)

### Calculs
- Température
- Entropie
- Pression
- Capacité calorifique
- +Graphiques

### Interface graphique
- Affichage des molécules selon l'état du système

# ÉCHÉANCIER
## Semaine 9 (09/03 - 15/03):
### (11/03) RAPPORT INTERIMAIRE
- Construire l'infrastrucutre de simu
- Modèle de l'eau à implémenter (2 semaines)
- Nosé-Hoover à implémenter (2-3 semaines)
- Interface graphique à implémenter (1-2 semaines)

## Semaine 10 (16/03 - 22/03):
- Modèle de l'eau implémenté 
- Interface graphique complétée
- Verlet à implémenter (1 semaine)
- Conception d'un notebook de test?

## Semaine 11 (23/03 - 29/03):
- Commencer à implémenter les tests de caractérisation
  - Capacité calorifique
  - Densité
  - Courbe P-T coexistence

## Semaine 12 (30/03 - 05/04):
- Infrastrucutre de simu complétée (optimiste)
- Implémenter le 2D
- Test + Résultats

## Semaine 13 (06/04 - 12/04):
### Optimisation
- Réécriture des étapes critiques du code en Julia.
- Utilisation de CuPy ou Cuda.jl
### Comparaison
- Ajout d'autres modèles d'eau (TIP4P/2005)
- Ajout d'autres thermostats

Commencer à préparer la présentation pour orale.
## Semaine 14 (13/04 - 19/04):
### (16/04) CAHIER DE PROJET
### (16/04) EVALUATION PAR LES PAIRS
### (19/04) DIAPOSITIVES PRESENTATION ORALE
Commencer rédaction du rapport final

## Semaine 15 (20/04 - 26/04):
### (20/04) (21/04) PRESENTATION ORALE

## Semaine 16 (27/04 - 03/05):
## Semaine 17 (04/05 - 10/05):
### (06/05) REMISE RAPPORT FINAL 