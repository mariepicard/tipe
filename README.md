# TIPE

## Description

Ce dossier contient mon projet de TIPE à l'issue de la prépa. Il s'agit de la reconnaissance automatique de cartes à jouer, en se privant de réseaux de neurones. La méthode doit être robuste à des perspectives et rotations importantes, mais ne traite que des cartes correctement éclairées, et non occultées. Il se concentre surtout sur l'optimisation de la détection des droites avec la transformation de Hough, nécessaire pour redresser l'image.
La partie de reconnaissance en elle-même est extrêmement simpliste et gagnerait certainement à exploiter du template-matching, ou bien des algorithmes de classification un peu plus évolués, comme les k plus proches voisins.
Le projet a été réalisé en Python afin de disposer d'un affichage facile. Il serait sans doute plus intéressant toutefois d'implémenter ceci en C pour bénéficier d'un temps de calcul plus restreint. 

---

This folder consists in my TIPE (final project), that I did at the end of my post-graduate preparatory class. The point is to recognize playing cards with simple tools (no neural network for instance). I tried to make it as invariant to rotation and perspective as possible, but it only works with well-lit and non-occluded cards. I mostly focused on optimizing line detection in the Hough transformation, which is necessary to correctly map the card.
The recognition aspect is somewhat simplistic, and could probably benefit from template-matching or other classification algorithms, such as the k nearest neighbours. 
This project was written in Python to have an easy access to plotting functions. It would be interesting to implement this in C, though, in order for it to be much quicker.

## Utilisation

La fonction principale est `analyse_carte`. Par défaut, elle se contente de renvoyer la valeur et l'enseigne identifiées. Les options `m_temps` et `affichage` permettent respectivement de renvoyer le détail du temps de calcul (i.e. le temps du filtre de Canny, de la transformation de Hough, de la reconstitution du symbole, d'identification de l'enseigne et de la valeur)  de surcroît, et d'afficher le processus de calcul de l'algorithme.

Exemple d'utilisation :

```python
#renvoie ('3', 'carreau')
analyse_carte("carreau3")

#renvoie ('3', 'carreau', '0.3264789581298828', '0.11625790596008301', '0.29645872116088867', '0.0017611980438232422', '0.041744232177734375')
analyse_carte("carreau3", m_temps = True)

#renvoie ('3', 'carreau') et affiche le processus
analyse_carte("carreau3", affichage = True)
```
