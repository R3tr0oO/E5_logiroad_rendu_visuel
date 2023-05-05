# ///////////////////////////////
#
# Dans cette version, l'étirement
# d'histogramme contient un bug
# - Applique l'effet uniquement sur
#   la dernière boîte englobante d'une
#   image.
#
# By : PEREZ K.
# Histogram Stretching by : CHATEAU T.
#
# modifications 03/05/2022
# Loïc ALIZON
#
# modifications 01/2023
# Maxime HERQUE
#
# ///////////////////////////////

# ===============================================
# ================ CONFIGURATION ================
# ===============================================
epaisseurRectangles = 5 # thickness of the borders of box
txtsize = 25 # Size of text inside textbox
tagbordersize = 2 # padding of tag box

#
# Take ID as input and configure the background,
# foreground colors and translation of the
# bounding box and his tag.
#
# @params defaut {String} - Contains the ID of damages
# @return {List}[String,String,String] Respectively: [BackgroundColour(B,G,R), Text, ForegroundColour]
#
# ===============================================
# ================ == PROGRAM == ================
# ===============================================

# Import of libs
import os, sys, logging, json, shutil, argparse, cv2, numpy as np, glob
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from skimage import data, color, img_as_ubyte, exposure
import pdb, re
# Logger configuration
logging.basicConfig(format='[%(levelname)s] %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

# Static variables
inputJsonList = [] # Constant
imageliste = [] # Constant
width, height = 0, 0 # Constant


#
# ARGUMENTS PARSER
#
parser = argparse.ArgumentParser()
parser.add_argument('--score', '-s', help='Afficher le score', action='store_true')
parser.add_argument('--classes', '-c', help='Afficher la classe (en anglais par défaut)', action='store_true')
parser.add_argument('--lang', '-l', type=str, help='Utilisez pour choisir la langue des classes (./lang_en.txt par défaut)')
parser.add_argument('--identifiant', '-id', help='Afficher identifiant', action='store_true')
parser.add_argument('--font', '-f', type=str, help='Changer la police d''écriture pour class & score (Chemin vers le fichier .TTF ou .OTF) (par défaut ./font.ttf)')
parser.add_argument('--inputjson', '-ij', type=str, help='Chemin vers le fichier json de Yolo (par défaut ./input_demo/*.json)')
parser.add_argument('--inputimgs', '-ii', type=str, help='Chemin vers (dossier) les images d''entrées (par défaut ./input_demo/images/*')
parser.add_argument('--outputimgs', '-oi', type=str, help='Chemin vers (dossier) la sortie (par défaut ./output)')
parser.add_argument('--video', '-v', type=int, help='Sortir une vidéo. Syntaxe: -v <IPS>')
parser.add_argument('--imagesempty', '-ie', help='Sortir les images sans modifications.', action='store_true')
parser.add_argument('--etirementhistogramme', '-eh', help="Permet un étirement d'histogramme", action='store_true')

args = parser.parse_args()

#
# This functions allows to configure the tag box and the language of the interface
#
# @params defaut {String} - Contains the ID of damages
# @return {List}[String,String,String] Respectively: [BackgroundColour[B,G,R], Text (in different languages), ForegroundColour]
#
def MiseAuPropreRectangle(defaut):

    content = []
    lefichier = ''
    
    if (args.classes == True):

        if (args.lang == None):
            logging.warning("Aucun argument, utilisation anglais")
            lefichier = 'lang_en.txt'
            with open(lefichier, "r", encoding='utf-8') as f: 
                for i in f.readlines():
                    defautcorrige = re.sub('\n',"", i)
                    content.append(defautcorrige)

        else:
            lefichier = args.lang
            logging.info("langue trouvée")
            with open(lefichier, 'r', encoding='utf8') as f: # sélectionne le dictionnaire selon l'argument choisi
                for i in f.readlines(): # se déplace dans le dictionnaire en fonction du numéro de ligne
                    defautcorrige = re.sub('\n',"", i) # retire tout les sauts de ligne du dictionnaire
                    content.append(defautcorrige) # intègre à une liste tout les éléments du dictionnaire

    print(content) # vérification console de l'éxactitude de la lange 
    
    #pdb.set_trace()

    if (defaut == "Arrachement pelade"):
        return [(142,129,80), content[0], "#fff"]
    if (defaut == "Reparation"):
        return [(42,175,225), content[1], "#000"]
    if (defaut == "Transversale"):
        return [(224,66,20), content[2], "#fff"]
    if (defaut == "Longitudinale"):
        return [(81,31,224), content[3], "#fff"]
    if (defaut == "Nid de poule"):
        return [(67,224,65), content[4], "#000"]
    if (defaut == "Faïencage"):
        return [(255,255,255), content[5], "#000"]
    else:
        return [(0,0,255), content[6], "#fff"]


#
# This functions allows to stretch histogram stretching
#
# @params im {NUMPY.NDARRAY} - Contains the input image
# @params sc {float} - scale applied to contrast
# @return ims {NUMPY.NDARRAY} - Contains the stretched image
#
def rgb2stretch(im,sc = 1.5):
    if (im.size != 0):
        # Convert to HSV
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        # histogram stretching
        imv = im_hsv[:,:,2]
        p2, p98 = np.percentile(im_hsv[:,:,2], (2, 98))
        imv = np.clip(imv,p2,p98)
        moy = np.mean((p2,p98))
        delta = p98 - p2
        ds = delta*sc
        np2 = np.clip(moy - ds,0,255)
        np98 =np.clip(moy + ds,0,255)
        #pdb.set_trace()
        a = float(np2 - np98) / float(p2 - p98)
        b = np98-a*p98
        imv = a*imv+b
        #imv = exposure.rescale_intensity(im_hsv[:,:,2], in_range=(p2, p98))
        im_hsv[:,:,2]=imv.astype(dtype = 'uint8')
        ims = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        #factor = 1 / 4.0
        #ims = ImageEnhance.Sharpness(ims).enhance(factor)
        return ims

# Top of the algorithm 
if (args.inputjson == None):
    logging.warning("Aucun argument pour localiser le fichier input, recherche dans ./input_demo/")
    for file in os.listdir(os.fsencode("./input_demo")):
        filename = os.fsdecode(file)
        if filename.endswith(".json"):
            lefichier = "./input_demo/" + filename
            logging.info(">> " + lefichier + " sera utilisé !")
            break
else:
    if (os.path.isfile(args.inputjson) == False):
        logging.error("Le fichier json " + args.inputjson + "n'existe pas." )
    else:
        lefichier = os.fsencode(args.inputjson)


# Check if input images folder exist
if (args.inputimgs == None):
    logging.warning("Aucun argument pour localiser le fichier input, recherche dans ./input_demo/images/")
    inputimages = "./input_demo/images/"
else:
    if (os.path.isdir(args.inputimgs) == False):
        logging.error = "Dossier d'image introuvable"
        exit()
    else:
        inputimages = args.inputimgs


# Check if output images folder exist
if (args.outputimgs == None):
    logging.warning("Aucun argument pour localiser le dossier output, utilisation de ./output/")
    outputimages = "./output/"
else:
    if (os.path.isdir(args.outputimgs) == False):
        logging.error = "Dossier d'image introuvable"
        exit()
    else:
        outputimages = args.outputimgs

# Beginning
with open(lefichier) as jsonFile:
    jsonObject = json.load(jsonFile)

    format_coco = False
    if('images' in jsonObject and 'categories' in jsonObject and 'annotations' in jsonObject):
        logging.info("Format Coco détecté, le format Coco ne possède pas de scores.")
        format_coco = True
    else:
        logging.info("Format Coco non détecté, traitement du format L²R Base.")

    logging.info("Traitement en cours...")

    ############# Format Coco #############
    if(format_coco):
        nbrimg = len(jsonObject['images'])
        noimg = 0
        lastpercent = 0
        for image in jsonObject['images']:
            noimg += 1
            if (args.video != None):
                imageliste.append(image)

            anns = [ann for ann in jsonObject['annotations'] if ann['image_id'] == image['id']]
            if(len(anns)==0):
                # SI PAS DE DÉTECTION on copie juste l'image
                if (args.imagesempty == True):
                    shutil.copy(inputimages + image, outputimages + image)
            else:
                # SI DÉTECTÉ on traite l'image
                print("============================")
                img = cv2.imread(inputimages + image['file_name'])
                lesRectangles = []
                width = image['width']
                height = image['height']
                
                for ann in anns:
                    x1 = 0 if ann['bbox'][0] <= 0 else ann['bbox'][0]   # Coordonnées du point en haut
                    y1 = 0 if ann['bbox'][1] <= 0 else ann['bbox'][1]   #  à gauche de la boîte
                    w = ann['bbox'][2]                                  # Largeur de la boîte
                    h = ann['bbox'][3]                                  # Hauteur de la boîte
                    x2 = width if x1 + w >= width else x1 + w           # Coordonnées calculées du point
                    y2 = height if y1 + h >= height else y1 + h         #  en bas à droite de la boîte
                    identifiant = ann[4]

                    y3 = y1 - txtsize                                   # Point en haut à gauche du texte

                    rst = MiseAuPropreRectangle(jsonObject['categories'][ann['category_id']]['name'])
                    lesRectangles.append([x1, y1, x2, y2, rst[0], rst[1], 0, y3, y1, rst[2], identifiant])
   
                    nowimg = round(noimg / nbrimg * 100)
                    if (lastpercent != nowimg):
                        logging.info("Avancement: " + str(nowimg) + "%")
                        lastpercent = nowimg
                for LeRectangle in lesRectangles:
                    if (args.classes == True or args.score == True):
                        text = ""
                        if (args.classes == True or args.score == True):
                            text = ""
                            if (args.classes == True and args.score == True and args.identifiant == True): 
                                text = str(LeRectangle[10]) + " | " + LeRectangle[5] + " | " + str(LeRectangle[6]) + "%"
                            elif (args.classes == True):
                                text = LeRectangle[5]
                            else:
                                text = str(LeRectangle[6]) + "%"
                            if (args.font == None):
                                fnt = ImageFont.truetype("./font.otf", txtsize)
                            else:
                                try:
                                    fnt = ImageFont.truetype(args.font)
                                except FileNotFoundError:
                                    logging.error("Police d'écriture introuvable...")
                                    exit()
                        # Bounding box tag generator
                        img_pil = Image.fromarray(img)
                        draw = ImageDraw.Draw(img_pil)
                        wtxt, htxt = draw.textsize(text, font=fnt)
                        xtxt = LeRectangle[0] + epaisseurRectangles / 2         # Abcisse du point en haut à gauche du texte (cas normal)
                        if xtxt + wtxt > width :                                # Si ça dépasse à droite de l'image, on décale le texte pour que la fin arrive au bout de l'image
                            xtxt = LeRectangle[2] - wtxt - epaisseurRectangles / 2
                        ytxt = (LeRectangle[8] - LeRectangle[7] - htxt)/2 + LeRectangle[7] - epaisseurRectangles    # Ordonée du point en haut à gauche du texte (cas normal)
                        if ytxt < 0:                                            # Si ça ne rentre pas au dessus de la boîte :
                            nouv = ytxt + LeRectangle[3] - LeRectangle[7]
                            if nouv + htxt <= height :                          # Si ça rentre on le colle en dessous
                                ytxt = nouv
                            else:                                               # Sinon on le descend au niveau du haut de la boîte
                                ytxt += htxt
                                if xtxt + LeRectangle[2] - LeRectangle[0] + wtxt <= width:  # Et si il rentre à droite de la boite on le met à droite
                                    xtxt += LeRectangle[2] - LeRectangle[0] + epaisseurRectangles / 2
                                elif xtxt - wtxt >= 0:                                      # Sinon si il rentre a gauche on le met a gauche, sinon on touche rien (et il sera dedans)
                                    xtxt -= wtxt       
                        draw.rectangle([xtxt-tagbordersize, ytxt-tagbordersize, xtxt+wtxt+tagbordersize, ytxt+htxt+tagbordersize], fill=LeRectangle[4], outline=LeRectangle[4], width=epaisseurRectangles)                          
                        draw.text((xtxt, ytxt), text, font=fnt, align='center', fill=LeRectangle[9])
                        img = np.array(img_pil)
                        # Histogram Stretching generator
                        # pdb.set_trace()
                    if ( args.etirementhistogramme == True):
                        rect = [round(LeRectangle[0]),round(LeRectangle[2]),round(LeRectangle[1]),round(LeRectangle[3])]
                        imROI = img[rect[2]:rect[3],rect[0]:rect[1],:]
                        toto = rgb2stretch(imROI)
                        if toto is not None:
                            img[rect[2]:rect[3],rect[0]:rect[1],:]=toto
                    cv2.rectangle(img, (round(LeRectangle[0]), round(LeRectangle[1])), (round(LeRectangle[2]), round(LeRectangle[3])), LeRectangle[4], epaisseurRectangles)
                # When finished, we save the image
                cv2.imwrite(outputimages + image['file_name'], img)

    ########### Format L²R Base ###########
    else:
        nbrimg = len(jsonObject)
        noimg = 0
        lastpercent = 0
        for image in jsonObject:
            noimg += 1
            if (args.video != None):
                imageliste.append(image)
            try:
                jsonObject[image][0]
            except IndexError:
                # IF NO DETECTION → We copy the image if option -ie is set
                if (args.imagesempty == True):
                    shutil.copy(inputimages + image, outputimages + image)
            else:
                # OTHERWISE we apply settings (score, class, histogram stretching...)
                img = cv2.imread(inputimages + image)
                lesRectangles = []
                height, width, c = img.shape
                for detections in jsonObject[image]:
                    #pdb.set_trace()
                    x = detections[2][0] / 608 * width 
                    y = detections[2][1] / 608 * height 
                    w = detections[2][2] / 608 * width 
                    h = detections[2][3] / 608 * height 
                    x1 = 0 if x-w/2 <= 0 else x-w/2
                    y1 = 0 if y-h/2 <= 0 else y-h/2
                    x2 = width if x1 + w >= width else x1 + w
                    y2 = height if y1 + h >= height else y1 + h
                    identifiant = detections[3]


                    y3 = y1 - txtsize
                    rst = MiseAuPropreRectangle(detections[0])
                    #pdb.set_trace()
                    
                    lesRectangles.append([x1, y1, x2, y2, rst[0], rst[1], round(float(detections[1])), y3, y1, rst[2], identifiant])
                    

                    # Percent step calculation
                    nowimg = round(noimg / nbrimg * 100)
                    if (lastpercent != nowimg):
                        logging.info("Avancement: " + str(nowimg) + "%")
                        lastpercent = nowimg
                # Bounding box generator
                for LeRectangle in lesRectangles:
                    if (args.classes == True or args.score == True):
                        text = ""
                        if (args.classes == True or args.score == True):
                            text = ""
                            if (args.classes == True and args.score == True and args.identifiant == True):
                                text = str(LeRectangle[10]) + " | " + LeRectangle[5] + " | " + str(LeRectangle[6]) + "%"  
                            elif ((args.classes == True or args.classesfrench)):
                                text = LeRectangle[5]
                            else:
                                text = str(LeRectangle[6]) + "%"
                            if (args.font == None):
                                fnt = ImageFont.truetype("./font.otf", txtsize)
                            else:
                                try:
                                    fnt = ImageFont.truetype(args.font)
                                except FileNotFoundError:
                                    logging.error("Police d'écriture introuvable...")
                                    exit()
                            
                        # Bounding box tag generator
                        img_pil = Image.fromarray(img)
                        draw = ImageDraw.Draw(img_pil)
                        wtxt, htxt = draw.textsize(text, font=fnt)
                        xtxt = LeRectangle[0] + epaisseurRectangles / 2
                        if xtxt + wtxt > width :
                            xtxt = LeRectangle[2] - wtxt - epaisseurRectangles / 2
                        ytxt = (LeRectangle[8] - LeRectangle[7] - htxt)/2 + LeRectangle[7] - epaisseurRectangles
                        if ytxt < 0:
                            nouv = ytxt + LeRectangle[3] - LeRectangle[7]
                            if nouv + htxt <= height :
                                ytxt = nouv
                            else:
                                ytxt += htxt
                                if xtxt + LeRectangle[2] - LeRectangle[0] + wtxt <= width:
                                    xtxt += LeRectangle[2] - LeRectangle[0]  + epaisseurRectangles / 2
                                elif xtxt - wtxt >= 0:
                                    xtxt -= wtxt    
                        draw.rectangle([xtxt-tagbordersize, ytxt-tagbordersize, xtxt+wtxt+tagbordersize, ytxt+htxt+tagbordersize], fill=LeRectangle[4], outline=LeRectangle[4], width=epaisseurRectangles)                          
                        draw.text((xtxt, ytxt), text, font=fnt, align='center', fill=LeRectangle[9])
                        img = np.array(img_pil)
                        # Histogram Stretching generator
                        # pdb.set_trace()
                    if ( args.etirementhistogramme == True):
                        rect = [round(LeRectangle[0]),round(LeRectangle[2]),round(LeRectangle[1]),round(LeRectangle[3])]
                        imROI = img[rect[2]:rect[3],rect[0]:rect[1],:]
                        toto = rgb2stretch(imROI)
                        if toto is not None:
                            img[rect[2]:rect[3],rect[0]:rect[1],:]=toto
                    cv2.rectangle(img, (round(LeRectangle[0]), round(LeRectangle[1])), (round(LeRectangle[2]), round(LeRectangle[3])), LeRectangle[4], epaisseurRectangles)
                # When finished, we save the image
                cv2.imwrite(outputimages + image, img)

    logging.info("Opération terminée.")
    jsonFile.close()

# Video generator 
if (args.video != None):
    logging.info("Démarrage du traitement pour génération de vidéo... (Sortie dans le dossier défini par le paramètre --outputimgs ou -oi)")
    
    img_array = []
    for file in os.listdir(os.fsencode(outputimages)):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            lefichier = outputimages + filename
            img = cv2.imread(lefichier)
            print(img)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
    
    logging.info("Génération de la vidéo...")
    out = cv2.VideoWriter(outputimages + 'outputvideo.mp4', cv2.VideoWriter_fourcc(*'MP4V'), args.video, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    logging.info("Opération vidéo terminée.")

    #### test
