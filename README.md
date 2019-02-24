# IA316_Kachouri_Veshchezerova
Project of the IA316 course

Ce dépot Git a pour objectif de répondre aux différentes parties du projet IA 316: 

    - Implémentation des 3 envrionnements 
    - Création d'une API fonctionnelle sur le 1er envrionnement.

1. 3ème Environnement: 

Nous avons testé plusieurs modèles : Siamese Network basique, Siamese Network avec covariates, puis on rajouté un calcul de similarité Cosinus entre les profils d'utilisateurs pour géréer les problèmes de cold start (nouveaux users lors des prédictions).
Par ailleurs, nous avons aussi testé un nouveau modèle se basant sur la FM et la WRAP loss puis nous avons fait une sorte de modèle hybride. Ces implémentations sont dans le notebook Third Environment 1.2 

2. Création d'une API:

L'API est fonctionnel en exécutant la commande docker-compe up --build . Si jamais il y a un souci, nous avons quand m^mee mis un notebook permettant de tester l'API et les deux méthodes implémentées
