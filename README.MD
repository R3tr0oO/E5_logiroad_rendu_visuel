
#  Rendu Visuel Logiroad

![Logiroad](https://i0.wp.com/logiroad.ai/wp-content/uploads/2021/12/cropped-logiroad-ai-logo-long.png?fit=1302,277&ssl=1)

*Un outil de rendu visuel des défauts de la chaussée.*

# Table des matières
1. [Installation](#installation)
2. [Technologies](#technologies)
3. [Fonctionnalités](#fonctionnalités)
				1. [Score](#score)
				2. [Identifiant](#identifiant)
				3. [Langue](#langue)
				4. [Classes](#classes)
				5. [Output](#output)
				6. [Input](#input)
				7. [Saturation](#saturation)
				8. [Combinaison](#combinaison)
				9. [Video](#video)
4. [Langages](#langages)


#  Installation


Utiliser le gestionnaire de librairies [pip](https://pip.pypa.io/en/stable/) pour installer les dépendances.

```bash

pip install PIL, skimage, re

```

#  Technologies

  

L'outil de rendu visuel est programmé en **Python3**, et utilise la librairie **Pillow2** ainsi que **Skimage** et **RegularExpressions**.

  
  

#  Fonctionnalités

  

L'outil de rendu visuel comporte plusieurs fonctionnalités.

  

##  Score

  

Utilisation:

```bash

python --score

```

Permet d'afficher le score des défaut de la chaussée.

  

##  Identifiant

  

Utilisation:

```bash

python --identifiant

```

Permet de suivre un même défaut sur une suite d'images.

  
  

##  Langue

  

Utilisation:

```bash

python --lang lang_es.txt

```

Permet de paramétrer le langage de l'affichage, en spécifiant le fichier de langue.

Les langues actuellement disponibles:

- Anglais

- Français

- Espagnol

  

##  Classes

  

Utilisation:

```bash

python --classes

```

Permet d'afficher le nom du défaut de la chaussée. Le langage par défaut est l'Anglais.

Pour changer la langue, il faut combiner avec le paramètre *lang*:

```bash

python --classes --lang lang_fr.txt

```

  

##  Output

  

Utilisation:

```bash

python --outputimgs ./output/

```

Permet de définir le fichier de sortie des images traitées.

  

##  Input

  

Utilisation:

```bash

python --inputimgs ./input/

```

Permet de définir le fichier stockant les images devant être traitées.

  

##  Saturation

  

Utilisation:

```bash

python --etirementhistogramme

```

Permet de saturer l'image à l'intérieur de la boite pour permettre une meilleur visibilité du défaut.

## Combinaison

Il est possible de combiner plusieurs paramètres. Pour avoir un rendu complet sur l'image traitée, il est possible d'entrer cette commande:
```bash
python --classes --lang lang_fr.txt --identifiant --score --etirementhistogramme
```

## Video

Il est possible de combiner plusieurs images afin d'en faire une vidéo. Il faut alors entrer cette commande:
```bash
python --video
```

# Langages

Il est possible de rajouter des langages. Pour cela, il faut simplement créer un fichier texte avec la nomenclature: *lang_(premières lettres de la langue).txt* .

Le fichier devra comporter les traductions dans l'ordre ci-dessous:
```
Arrachement Pelade
Réparation
Transversale
Longitudinale
Nid de Poule
Faïencage
Inconnue
```
Le fichier sera placé à la racines du répertoire ou dans un répertoire dédié.