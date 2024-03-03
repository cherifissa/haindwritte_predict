import cv2
import numpy as np
import matplotlib.pyplot as plot
import os
# Charger l'image
image = cv2.imread('image.tif')

# Afficher l'image originale
#cv2.imshow('Image originale', image)
#cv2.waitKey(0)

# Conversion en niveaux de gris
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Afficher l'image en niveaux de gris
#cv2.imshow('Image en niveaux de gris', image_gray)
#cv2.waitKey(0)

# Histogramme
H = cv2.calcHist([image_gray], [0], None, [256], [0,256])
plot.hist(H)
# plot.show()

# Binarisation
_, image_binaire = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Afficher l'image binaire
#cv2.imshow('Image binaire', image_binaire)
#cv2.waitKey(0)

# Pretraitement et correction
im_bw0 = cv2.threshold(image_binaire, 127, 255, cv2.THRESH_BINARY)[1]

kernel = np.ones((19,19),np.uint8)
im_bw1 = cv2.dilate(im_bw0, kernel)
im_bw1 = cv2.erode(im_bw1, kernel)

# Remove border pixels
im_bw2 = im_bw1.copy()
im_bw2[0,:] = 0
im_bw2[-1,:] = 0
im_bw2[:,0] = 0
im_bw2[:,-1] = 0

im_bw3 = cv2.threshold(im_bw2, 0, 255, cv2.THRESH_BINARY)[1]

#cv2.imshow('Image binaire après traitement', im_bw3)
#cv2.waitKey(0)

# Extraction
contours, _ = cv2.findContours(im_bw3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

dossier_dest = "pred"

# Créer le dossier de destination s'il n'existe pas
if not os.path.exists(dossier_dest):
    os.makedirs(dossier_dest)
image_num = 1

# Parcourir tous les contours détectés
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    caractere_recadre = im_bw3[y:y+h, x:x+w]

    # Ajouter une bordure de 15 pixels noirs à tous les côtés de l'image
    caractere_recadre_bordered = cv2.copyMakeBorder(caractere_recadre, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Redimensionner l'image à 28x28 pixels
    caractere_recadre_resized = cv2.resize(caractere_recadre_bordered, (28, 28))

    # Générer un nom de fichier unique
    nom_fichier = f"{image_num}.png"
    
    # Incrémenter le compteur pour le prochain fichier
    image_num += 1
    
    # Chemin complet du fichier
    chemin_fichier = os.path.join(dossier_dest, nom_fichier)

    # Enregistrer l'image redimensionnée dans le dossier de destination
    cv2.imwrite(chemin_fichier, caractere_recadre_resized)

    #cv2.imshow('Caractère recadré', caractere_recadre)
    #cv2.waitKey(0)

