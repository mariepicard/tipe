import numpy as np
from PIL import Image
from scipy import ndimage
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time

##image en niveaux de gris afin de pouvoir calculer le gradient
def rgb2gray(img):
    """met l'image (représentée par un tableau numpy) en nuance de gris"""
    new = []
    for ligne in img:
        lg = []
        for p in ligne:
            lg.append((int(p[0]) + int(p[1]) + int(p[2]))//3)
        new.append(lg)
    return np.array(new)

##filtre de canny pour détecter les contours
def gradient(img):
    """calcule le gradient d'une image en nuances de gris, avec la méthode de sobel"""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    
    return G

def canny(gr, b=True):
    """
    Applique le filtre de Canny sur une image en noir et blanc, afin de déterminer les points de contour

    Processus:
    -> filtre de gauss pour lisser
    -> pas de suppression des non-maximas (bruite l'image)
    -> double seuillage

    b permet de choisir à quel point les seuils seront haut. Par défaut, le seuil haut vaut 90% de la valeur max, et le seuil bas, 63%.
    """
    M = len(gr)
    N = len(gr[0])
    pile = []
    if b:
        smax = gr.max() * 0.9
        smin = smax*0.7
    else:
        smax = gr.max() * 0.6
        smin = smax*0.2
    new = np.array([[0.5 for _ in range (N)] for _ in range (M)])
    pile.append((0, False))#pile des pixels et état (contour ou non)
    while pile != []:
        p, cont = pile.pop()
        if new[p//N][p%N] != 1:
            if (gr[p//N][p%N] >= smax) or ((gr[p//N][p%N] >= smin) and cont):
                new[p//N][p%N] = 1
                if p%N != N - 1:
                    pile.append((p + 1, True))
                if p%N != 0:
                    pile.append((p - 1, True))
                if p//N != 0:
                    pile.append((p - N, True))
                if p//N != M - 1:
                    pile.append((p + N, True))
            elif new[p//N][p%N] != 0:
                new[p//N][p%N] = 0
                if p%N != N - 1:
                    pile.append((p + 1, False))
                if p%N != 0:
                    pile.append((p - 1, False))
                if p//N != 0:
                    pile.append((p - N, False))
                if p//N != M - 1:
                    pile.append((p + N, False))
    return new

def contour(img_canny):
    """récupère la liste des points de contour d'une image en noir et blanc"""
    l = []
    for i in range(len(img_canny)):
        for j in range(len(img_canny[0])):
            if img_canny[i][j] == 1:
                l.append((j, i))
    return l

def axy(bimg, x, y):
    """fonction auxiliaire de thinning : compte le nombre de séries de 01 autour du pixel central"""
    a = 0
    v = [(y+1, x), (y+1, x+1), (y, x+1), (y-1, x+1), (y-1, x), (y-1, x-1), (y, x-1), (y+1, x-1)]
    for i in range(8):
        if bimg[v[i][0]][v[i][1]] == 0 and bimg[v[(i+1)%8][0]][v[(i+1)%8][1]] == 1:
            a+=1
    return a

def thinning(bimg, contour):
    """
    Entrées : bimg est le tableau numpy représentant l'image et contour ses points de contour
    Réseultat : amincit les brods d'une image bianire grâce à l'algorithme d'amincissement de zhang-suen
        -> conserve la connexité des pixels et la forme globale
        -> bimg est modifié en place
    Sortie : la liste des points de coontour après amincissement

    Attention : L'algorihme étant récursif, pour un motif trop épais, on peut s'attendre à un dépassement du nombre maximal d'appels
    """
    c = 0 #nombre de modifications
    h = len(bimg)
    l = len(bimg[0])
    m = np.zeros((h, l)) #matrice des modifications
    skel = [] #liste des points de contour conservés
    #première passe pour modifier les pixels au sud-est et les coins
    for (x, y) in contour:
        bij = bimg[y-1][x] + bimg[y-1][x-1] + bimg[y-1][x+1] + bimg[y][x-1] + bimg[y][x+1] + bimg[y+1][x-1] + bimg[y+1][x] + bimg[y+1][x+1]
        cij = bimg[y-1][x] * bimg[y][x+1] * bimg[y+1][x]
        dij = bimg[y][x-1] * bimg[y][x+1] * bimg[y+1][x]
        if bij >= 2 and bij<=6 and cij == 0 and dij == 0:
            aij = axy(bimg, x, y)
            if aij == 1:
                c += 1
                m[y][x] = 1
            else:
                skel.append((x, y))
        else:
            skel.append((x, y))
    if c == 0: #cas d'arrêt : pas de modification
        return skel
    bimg = bimg - m
    skel2 = []
    c = 0
    m = np.zeros((h, l))
    #deuxième passe pour supprimer les pixels au nord-ouest et les coins
    for (x, y) in skel:
        bij = bimg[y-1][x] + bimg[y-1][x-1] + bimg[y-1][x+1] + bimg[y][x-1] + bimg[y][x+1] + bimg[y+1][x-1] + bimg[y+1][x] + bimg[y+1][x+1]
        cij = bimg[y][x-1] * bimg[y][x+1] * bimg[y-1][x]
        dij = bimg[y][x-1] * bimg[y+1][x] * bimg[y-1][x]
        if bij >= 2 and bij<=6 and cij == 0 and dij == 0:
            aij = axy(bimg, x, y)
            if aij == 1:
                c += 1
                m[y][x] = 1
            else:
                skel2.append((x,y))
        else:
            skel2.append((x, y))
    if c == 0:
        return skel2
    bimg = bimg - m
    return thinning(bimg, skel2)

## linking and subdivision : on détecte des clusters de pixels à peu près colinéaires     
                       
def is_valid(p, img):
    """teste si la position d'un pixel est valide"""
    xp = p[0]
    yp = p[1]
    return xp >= 0 and xp<len(img[0]) and yp>=0 and yp<=len(img)

def pnext(p, img):
    """renvoie le pixel suivant d'une composante connexe de contour"""
    xp = p[0]
    yp = p[1]
    if (is_valid((xp+1, yp), img) and img[yp][xp+1] == 1):
        return (xp+1, yp)
    if (is_valid((xp-1, yp), img) and img[yp][xp-1] == 1):
        return (xp-1, yp)
    if (is_valid((xp, yp+1), img) and img[yp+1][xp] == 1):
        return (xp, yp+1)
    if (is_valid((xp, yp-1), img) and img[yp-1][xp] == 1):
        return (xp, yp-1)
    if (is_valid((xp+1, yp+1), img) and img[yp+1][xp+1] == 1):
        return (xp+1, yp+1)
    if (is_valid((xp-1, yp+1), img) and img[yp+1][xp-1] == 1):
        return (xp-1, yp+1)
    if (is_valid((xp+1, yp-1), img) and img[yp-1][xp+1] == 1):
        return (xp+1, yp-1)
    if (is_valid((xp-1, yp-1), img) and img[yp-1][xp-1] == 1):
        return (xp-1, yp-1)
    return (-1, -1)


def linking(img, pref):
    """
    Entrées :
        img : image binaire qui représente les bords
        pref: pixel initial qui fait partie du bord
    Résultat :
        chaîne les pixels voisins dans une liste, i.e produit une liste de pixels adjacents qui forment une composante connexe
    Sortie :
        liste correspondant à la chaîne. 2 pixels adjacents dans la liste le sont aussi sur l'image
    Pour limiter le temps de calucl, on utilise 2 piles
    """
    S = []
    S2 = []
    p = pref
    while is_valid(p, img):#on part dans un premier sens
        S.append(p)
        img[p[1]][p[0]] = 0
        p = pnext(p, img)
    p = pnext(pref, img)
    while is_valid(p, img): #puis dans l'autre
        S2.append(p)
        img[p[1]][p[0]] = 0
        p = pnext(p, img)
    S2.reverse()
    return S2+S


def subdivision(lst):
    """
    Entrées:
        lst est une liste et représente un ensemble de pixels de contour adjacents.
    Résultat :
        subdivise lst en segment approximativement colinéaires
    Sortie :
        liste de tuple (seg, pertinence) où seg est l'ensemble des points d'un des segments colinéaires et pertinence le rapport de la distance entre les points extrêmaux de la chaîne sur la déviation maximale
        
    Fonctionnement:
        On calcule récursivement des clusters de pixels de lst qui sont approximativement colinéaires en déterminant le ratio taille de la ligne/déviation maximale
        Notons qu'on peut modifier la condition d'arrêt par len(lst)/déviation maximale, ce qui assurera une complexité en O(n*log(n)) avec n = len(lst) (cf Mater's theorem)
        Dans le cas des cartes à jouer, le fonctionnement n'est pas vraiment modifié, d'où le choix de garder cette variante
    """
    if len(lst) <= 2: #cluster inintéressant
        return []
    (xa, ya) = lst[0] #pixels extrêmaux
    (xb, yb) = lst[-1]
    ab = np.sqrt((xb - xa)**2 + (yb - ya)**2)

    #calcul de la déviation maximale d'un pixel de lst par rapport à la droite passant par ces deux pixels extrêmaux a et b
    m = 0
    idx = 0
    for i in range (1, len(lst) - 1):
        (x, y) = lst[i]
        d = abs((x - xa)*(yb - ya) - (xb - xa)*(y-ya))/ab
        if d > m:
            m = d
            idx = i

    if m == 0 : #déjà un segment
        return [(lst, np.Infinity)]
    
    r = ab/m #rapport de pertinence
    if len(lst) <= 5 or r >= 150: #pas de subdivision nécessaire
        return [(lst, r)]
    
    s1 = subdivision(lst[:idx]) #calcul récursif sur les 2 sous-chaînes séparées au pixel de déviation maximal
    s2 = subdivision(lst[idx:])
    for (s, rs) in s1: #un des sous-segments calculé est plus pertinent
        if rs > r:
            return s1 + s2
    for (s, rs) in s1:
        if rs > r:
            return s1 + s2
    return [(lst, r)] #aucun sous-segment calculé n'est plus pertinent
        
def clu(lst):
    """renvoie la liste des clusters de signification >= 8 (i.e le ratio longueur/déviation >= 8)"""
    ls = subdivision(lst)
    l = []
    for (s, rs) in ls:
        if rs >= 8 :
            l.append(s)
    return l

def dfs_clusters(img):
    """calcule tous les clusters de points en effectuant un parcours en profondeur sur chaque composante connexe de points de contour"""
    l_clusters = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 1:
                lij = linking(img, (j, i))
                l_clusters = l_clusters + clu(lij)
    return l_clusters

def print_clu (img, clu, col = [0,0,0]):
    """colore les pixels des cluters de la liste clu. On peut éventuellement imposer la couleur"""
    a = len(clu)
    for i in range (len(clu)):
        if col != [0,0,0]:
            for (x, y) in clu[i]:
                img[y][x] = col
        elif i%5 == 0:
            for (x, y) in clu[i]:
                img[y][x] = [1/2, 1/2, 0]
        elif i%5 == 1:
            for (x, y) in clu[i]:
                img[y][x] = [0, 1/2, 0]
        elif i%5 == 2:
            for (x, y) in clu[i]:
                img[y][x] = [0, 2/3, 1/3]
        elif i%5 == 3:
            for (x, y) in clu[i]:
                img[y][x] = [1/2, 0, 1/2]
        else:
            for (x, y) in clu[i]:
                img[y][x] = [0,0,2/3]

## calcul des noyaux gaussiens
def g_kernels(clusters, h, l):
    """
    Entrées :
        clusters est la liste des clusters de l'mage, obtenue grâce à un appel à dfs_clusters
        h et l sont des entiers correspondant aux dimensions de l'image
    Résultat :
        calcule les paramètres des gaussiennes bi-dimensionnelles qui serviront à voter dans l'espace de Hough
    Sortie :
        renvoie un tableau numpy de la liste des paramètres de la gaussienne, représentés sous forme de tuple contenant
            * rho, la distance de la droite à l'origine
            * theta, l'angle réalisant cette distance
            * sigma_r2 : la variance de rho
            * sigma_t2 : la variance de theta
            * cov_rt : la covariance entre rho et theta
    Le processus du calcul correspond à celui détaillé par Fernandes et Oliveira
    Lors du calcul, on porte une attention toute particulière dans le cas d'une droite verticale
    """
    origine = np.array([l//2, h//2]) #centre de l'image
    l_param = [] #liste des paramètres de chaque cluster
    
    for cl in clusters:
        n = len(cl)
        s = np.array(cl) - origine
        pm = np.mean(s, axis=0) #barycentre
        s = s - pm #on centre le cluster
        m = np.dot(np.transpose(s), s)
        vp, vect = npl.eig(m)
        if vp[0] > vp[1]:
            u = vect[:,0] #base orthonormée de vecteurs propres, adaptée à l'ellipse
            #notons que dans le cas où la subdivision est mal effectuée, la base peut être mal définie, si l'ellipse est un cercle
            #ce cas est toutefois statistiquement presque impossible
            v = vect[:,1]
        else:
            u = vect[:,1]
            v = vect[:,0]
        if v[1] < 0 :
            v = -v #pour la compatibilité avec l'équation, y_v >= 0

        #calcul des paramètres des noyaux gaussiens
        rho = int(np.dot(v, pm))
        theta = int(180*np.arccos(v[0])/np.pi)
        a = np.dot(s, u)
        sigma_m2 = 1/np.dot(a, a)
        sigma_b2 = 1/n
        #propagation d'incertitude
        if v[0] == 1: #droite verticale
            jacobian = np.array([(-np.dot(u, pm), 1), (1, 0)])
        else:
            jacobian = np.array([(-np.dot(u, pm), 1), (u[0]/abs(u[0]), 0)])
        M = np.dot(np.dot(jacobian, np.array([(sigma_m2, 0), (0, sigma_b2)])), np.transpose(jacobian))
        sigma_rho2 = 4*M[0][0]
        if M[1][1] == 0 :
            sigma_theta2 = 0.4
        else :
            sigma_theta2 = 4*M[1][1]
        cov_rhotheta = M[0][1]
        #cas de la droite verticale (covariance nulle)
        if v[0] == 0:
            cov_rhotheta = 0
        l_param.append((rho, theta, sigma_rho2, sigma_theta2, cov_rhotheta))
        
    return np.array(l_param)

def G(param_k, rho, theta):
    """gaussienne qui donne le coefficient de vote pour chaque cluster"""
    r, th, sigma_r2, sigma_t2, cov_rt = param_k
    r2 = cov_rt**2 / (sigma_r2*sigma_t2)
    z = (rho-r)**2/sigma_r2 - 2*cov_rt*(rho-r)*(theta-th)/(sigma_r2*sigma_t2) + (theta-th)**2/sigma_t2
    return np.exp(-z/(2*(1-r2)))/(2*np.pi*np.sqrt(sigma_r2*sigma_t2*(1-r2)))

def height_c(param_k):
    """hauteur d'une ellipse gaussienne"""
    rho, theta, sigma_rho2, sigma_theta2, cov_rhotheta = param_k
    r2 = cov_rhotheta**2 / (sigma_rho2*sigma_theta2)
    return 1/(2*np.pi*np.sqrt(sigma_rho2*sigma_theta2*(1-r2)))


def vote(votes, rho0, theta0, delta_rho, delta_theta, gs, param_k, R):
    """
    Entrées :
        votes est un tableau numpy qui correspond à l'espce de Hough dans lequel on vote
        rho0, theta0 les coordonnées de la droite pour laquelle on doit commencer par voter
        delta_rho, delta_theta les pas de discrétisation
        gs un coefficient pour s'assurer que les votes seront entiers
        param_k un tuple des paramètres de la gaussienne que l'on utilise pour voter
        R le rayon de l'image
    Résultat :
        procédure de votes pour chaque cluster en uilisant une répartition gaussienne définie par param_k
    Note :
        La gaussienne est divisée en 4 cadrans pour simplifier le vote. On se montre particulièrement attentif aux changements éventuels de cadrans
    """
    rho, theta, sigma_rho2, sigma_theta2, cov_rhotheta = param_k
    theta_j = theta0
    loop = True
    while loop: #la boucle s'interrompt quand les valeurs de la gaussienne sont trop faibles, et qu'aucun vote n'est ajouté
        if theta_j < 0 or theta_j > 180: #changement de cadran
            delta_rho *= -1
            rho *= -1
            cov_rhotheta *= -1
            rho0 *= -1
            if theta_j < 0:
                theta += 180 - delta_theta
                theta_j = 180 - delta_theta
            else:
                theta += delta_theta - 180
                theta_j = 0
        
        rho_j = rho0
        g = round(gs*G(param_k, rho_j, theta_j))
        c = 0
        while g > 0 and -R <= rho_j <= R:
            c += 1
            votes[int(rho_j)][int(theta_j)] += g
            rho_j += delta_rho
            g = round(gs*G(param_k, rho_j, theta_j))
        if c == 0:
            loop = False
        theta_j += delta_theta

def hough_amelioree(clusters, h, l):
    """
    Transformation de Hough décrite par Fernandes et Oliveira

    Entrées :
        clusters la liste des clusters, obtenue par dfs_clusters
        h et l sont les dimensions de l'image qu'on traite
    Résultat :
        calcule les droites principales du contour de l'image traitée
    Sortie :
        Renvoie la liste des droites principales, représentées par un tuple (a, b, c) telle que la droite vérifie l'équation cartésienne a*x + b*y + c = 0
    """
    R = int(np.sqrt(h**2 + l**2)//2) + 1
    delta = 1
    parametres = g_kernels(clusters, h, l)
    #suppression des clusters trop bas
    hmax = 0
    for param in parametres:
        if height_c(param) >= hmax:
            hmax = height_c(param)
    ncl = []
    cl = []
    for i in range(len(parametres)):
        if height_c(parametres[i]) >= hmax * 0.0005:
            ncl.append(parametres[i])
            cl.append(clusters[i])
    parametres = np.array(ncl)
    #seuil gmin pour voter avec des entiers
    gmin = 1
    for (rho, theta, sigma_rho2, sigma_theta2, cov_rhotheta) in parametres:
        M = np.array([(sigma_rho2, cov_rhotheta), (cov_rhotheta, sigma_theta2)])
        vp, vect = npl.eig(M)
        if vp[0] > vp[1]:
            lambda_w = vp[1]
            w = vect[:,1]
        else:
            lambda_w = vp[0]
            w = vect[:,0]
        gmin = min(gmin, G((rho, theta, sigma_rho2, sigma_theta2, cov_rhotheta), rho + w[0]*np.sqrt(lambda_w), theta + w[1]*np.sqrt(lambda_w)))
    votes = np.zeros((2*R + 2, 180))
    gs = max(1 / gmin, 1)

    #vote dans chaque cadran, pour chaque cluster
    for param_k in parametres:
        (rho, theta, sigma_rho2, sigma_theta2, cov_rhotheta) = param_k
        vote(votes, rho, theta, delta, delta, gs, param_k, R)
        vote(votes, rho, theta - delta, delta, -delta, gs, param_k, R)
        vote(votes, rho - delta, theta - delta, -delta, -delta, gs, param_k, R)
        vote(votes, rho - delta, theta, -delta, delta, gs, param_k, R)


    #détection des 4 premiers pics dans l'espace de Hough
    votes = ndimage.gaussian_filter(votes, 1.5)
    lst = ((votes >= 1).nonzero())
    l_triee = ((np.transpose(lst))[np.argsort(votes[lst])])[::-1]
    vus = np.zeros((2*R +1, 180))
    pics = []
    i = 0
    while i<len(l_triee) and len(pics)<4:
        idx = l_triee[i]
        r, theta = idx[0], idx[1]
        vus[r][theta] = 1
        if theta==0:
            aij = vus[r][1] + vus[r+1][0] + vus[r+1][1] + vus[-r][179] + vus[-r+1][179] + vus[r-1][0] + vus[-r-1][179] + vus[r-1][1]
        elif theta==179:
            aij = vus[r][178] + vus[r+1][179] + vus[r-1][179] + vus[r+1][178] + vus[r-1][178] + vus[-r][0] + vus[-r-1][0] + vus[-r+1][0]
        else:
            aij = vus[r][theta + 1] + vus[r][theta-1] + vus[r-1][theta] + vus[r-1][theta - 1] + vus[r-1][theta + 1] + vus[r+1][theta] + vus[r+1][theta-1] + vus[r+1][theta+1]
        if aij == 0:
            if r > R+1:
                r -= 2*R +1
            a = np.cos(theta*np.pi/180)
            b = np.sin(theta*np.pi/180)
            pics.append((a, b, -(r + a*(l//2) + b*(h//2))))
        i += 1
    return pics


##à partir des équations des droites, on détermibe les 4 coins de la carte
def is_possible(lignes, n, h, l):
    """indique si la combinaison est possible, sachant que la droite n est celle représentant le bord opposé à celui 0 en renvoyant la liste des coins:
    Si la liste est vide cela signifie que la combinaison est impossible"""
    a,b,c = lignes[0]
    a2, b2, c2 = lignes[n]
    coins = []
    possible = True
    for i in range(1,4):
        if i != n:
            e,f,d = lignes[i]
            if e*b == f*a or e*b2 == f*a2:
                possible=False
            elif f==0:
                coin1 = (int(-d/e), int((a*d/e - c)/b))
                coin2 = (int(-d/e), int((a2*d/e - c2)/b2))
                if coin1[0]<0 or coin1[0]>l or coin1[1]<0 or coin1[1]>h or coin2[0]<0 or coin2[0]>l or coin2[1]<0 or coin2[1]>h:
                    possible = False
                coins.append(coin1)
                coins.append(coin2)
            else:
                x1 = (f*c - b*d)/(e*b - f*a)
                x2 = (f*c2 - b2*d)/(e*b2 - f*a2)
                coin1 = (int(x1),int(-(e*x1 + d)/f))
                coin2 = (int(x2),int(-(e*x2 + d)/f))
                if coin1[0]<0 or coin1[0]>l or coin1[1]<0 or coin1[1]>h or coin2[0]<0 or coin2[0]>l or coin2[1]<0 or coin2[1]>h:
                    possible = False
                coins.append(coin1)
                coins.append(coin2)
    if possible:
        return coins
    else:
        return []

def coins(lignes, img):
    """
    Entrées :
        lignes est la liste des droites détectées par la transformation de Hough
        img est l'image traitée
    Résultat:
        détecte les coins de la carte sur l'image
    Sortie :
        liste des coordonnées des 4 points"""
    assert (len(lignes) == 4), "Trop peu de droites ont été détectées"

    for i in range (1, 4):
        coins = is_possible(lignes, i, len(img), len(img[0]))
        if coins != []:
            return coins

#on reconstitue la carte (on la redresse en calculant les coordonnées dans l'espace de chacun de ses points)
def penchee(symbole, p, test = True):
    """renvoie le pixel coloré de symbole le plus près du pixel demandé. S'il n'y en a pas, c'est que la carte a été redressée dans le mauvais sens"""
    #pas de parcours en largeur! effectuer une recherche sur la ligne de coordonnées! plus rapide et vrai
    cherche = True
    i = 0
    j = 0
    while cherche and -5 <= j <= 5:
        while 0 <= i + p[0] < len(symbole[0]) and cherche:
            if symbole[p[1] + j][p[0] + i] == 1:
                cherche = False
            else :
                if i < 0:
                    i = -i
                else:
                    i = -i-1
        if cherche and j < 0:
            j = - j
        elif cherche :
            j = - j - 1
    if cherche:
        return (-1, -1)
    return (p[0] + i, p[1] + j)
    
def recup_motif(symbole, p):
    """récupère le motif dans lequel p est inscrit"""
##    (x, y) = penchee(symbole, p, False)
    (x, y) = p
    if x == -1:
        return (-1, -1, -1, -1)
    vus = np.zeros((len(symbole), len(symbole[0])))
    pile = [(x, y)]
    xmin, xmax, ymin, ymax = x, x, y, y
    while pile != []:#recherche des coordonnées extrémales avec un parcours en profondeur de la zone connexe
        (x, y) = pile.pop()
        vus[y][x] = 1
        if symbole[y][x] == 1:
            if x > xmax:
                xmax = x
            elif x < xmin:
                xmin = x
            if y > ymax:
                ymax = y
            elif y < ymin:
                ymin = y
            if x < len(symbole[0])-1 and vus[y][x+1] == 0:
                pile.append((x+1, y))
            if x > 0 and vus[y][x-1] == 0:
                pile.append((x-1, y))
            if y > 0 and vus[y-1][x] == 0:
                pile.append((x, y-1))
            if y < len(symbole)-1 and vus[y+1][x] == 0:
                pile.append((x, y+1))
    return (xmin, xmax, ymin, ymax)


def thresholding(carte):
    """seuillage adapté d'un motif: il favorise le rouge pour pouvoir repérer les motifs rouges, et limiter le nombre de points de couleurs aberrantes"""
    mr = np.mean(carte[:,:,0])
    mg = np.mean(carte[:,:,1])
    mb = np.mean(carte[:,:,2])
    new = np.zeros((len(carte), len(carte[0]), 3))
    for i in range(len(carte)):
        for j in range(len(carte[0])):
            if carte[i][j][0] >= mr*0.85:
                if carte[i][j][1] <= mg*0.95 and carte[i][j][2] <= mb*0.95:
                    new[i][j] = [1,0,0]
                else:
                    new[i][j] = [1,1,1]
    return new

def m (coins):
    """
    Entrée:
        liste des coins de la carte
    Sortie :
        renvoie une matrice 11*12 dont le noyau correspond à la droite des possibilités des coordonnées des 4 coins de la carte en 3D

    Correction :
        la matrice est de rang 11 tant que les points ne sont pas alignés

    Note :
        L'espace 3D est défini par:
            * Centre optique : (0,0,0)
            * Plan optique : z=100
    """
    [(XA, YA), (XB, YB), (XD, YD), (XC, YC)] = coins 
    X = [XA, XB, XC, XD]
    Y = [YA, YB, YC, YD]
    M = np.zeros ((11,12))
    for b in range (4) :
        M[2*b,3*b] = 100
        M[2*b,3*b+2] = -X[b]
        M[2*b+1,3*b+1] = 100
        M[2*b+1,3*b+2] = -Y[b]
        for k in range (3) :
            M[k+8,k+3*b] = (-1)**(b+1)
    return M

def recreate_image(A, bc, bl, img, haut, larg):
    """
    Entrées :
        A est le coin sur lequel on recrée le motif
        bc est la liste des coordonnées du vecteur bord court
        bl, du vecteur bord long
        img correspond à l'image initiale, chez laquelle on récupère la couleur des pixels
        hau et larg sont les dimensions de l'image qu'on veut créer
    Sortie :
        un tableau numpy aux dimensions haut et larg, modifié pour contenir le motif dans le coin A

    Resultat :
        recrée le symbole de la carte à partir des informations précédentes
    """
    [xI, yI, zI] = A + 3*bc/55 + bl/42
    [xJ, yJ, zJ] = A + bl/42 + 13*bc/55
    [xK, yK, zK] = A + 4*bl/21 + 3*bc/55
    I = np.array([xI, yI, zI])
    IJ = np.array([xJ - xI, yJ - yI, zJ - zI])
    IK = np.array([xK - xI, yK - yI, zK - zI])
    im = np.zeros ((haut, larg, 3), dtype=int)
    for i in range (haut) :
        for j in range (larg) :
            [x, y, z] = I + IK*i/haut + IJ*j/larg
            im[i,j] = img[int(100*y/z), int(100*x/z)]
    return im

def symbole(img):
    """
    Entrée : img est le coin redressé
    Sortie : liste des coordonnées des rectangles minimaux contenant le chiffre et la couleur
    """
    bords = canny(gradient(rgb2gray(img)), False)
    return [recup_motif(bords, penchee(bords, (50, 105))), recup_motif(bords, penchee(bords, (50, 50)))]

def reconstitue_rectangle (coins, larg, img) :
    """
    Entrées :
        coins est la liste des coordonnées 2D des coins de la carte sur limage
        larg la largeur du motif qu'on veut reconstruire
        img la photographie de la carte qu'on traite
    Résultat:
        on reconstitue le rectangle du symbole et chiffre grâce aux coordonnées dans l'espace des coins de la carte
    Sortie :
        tuple contenant le coin redressé, la valeur et l'enseigne visible dans ce coin
    """
    lst_coins = [(a*15, b*15) for (a,b) in coins]
    haut = int(1.4*larg)
    M = m(lst_coins)
    xC, yC, zC, xD, yD, zD, xA, yA, zA, xB, yB, zB = np.reshape(spl.null_space(M),12)*100
    A = np.array ([xA,yA,zA])
    AB = np.array ([xB-xA,yB-yA,zB-zA])
    AD = np.array ([xD-xA,yD-yA,zD-zA])
    #on détermine quel est le bord long et lequel est court
    #on calcule les distances entre les points sur l'image
    dAB = np.sqrt((coins[0][0] - coins[1][0])**2 + (coins[0][1] - coins[1][1])**2)
    dAD = np.sqrt((coins[0][0] - coins[2][0])**2 + (coins[0][1] - coins[2][1])**2)
    dBC = np.sqrt((coins[3][0] - coins[1][0])**2 + (coins[3][1] - coins[1][1])**2)
    dCD = np.sqrt((coins[2][0] - coins[3][0])**2 + (coins[2][1] - coins[3][1])**2)
    chg_sg = False
    if min(dAB, dCD) > min(dBC, dAD):
        bl, bc = AB, AD #bord long, bord court
    else:
        bl, bc = AD, AB
    if bl[1]*bc[0] < 0:
        chg_sg = True
        A = A + bc
        bc = -bc
    #on calcule les coordonnées du rectangle dans lequel se trouvent le chiffre et le symbole
    im = recreate_image(A, bc, bl, img, haut, larg)
    [(xmine, xmaxe, ymine, ymaxe), (xminv, xmaxv, yminv, ymaxv)] = symbole(im)
    if xmine == -1 or xminv == -1: #on a échangé bord court et bord long
        if chg_sg:
            bc = -bc
            A = A - bc
        bc, bl = bl, bc
        if bc[1]*bc[0] < 0:
            A = A + bc
            bc = -bc
        im = recreate_image(A, bc, bl, img, haut, larg)
        [(xmine, xmaxe, ymine, ymaxe), (xminv, xmaxv, yminv, ymaxv)] = symbole(im)
    s = thresholding(im)
    ch = s[yminv - 1: ymaxv +2, xminv-1:xmaxv + 2]
    coul = s[ymine - 1: ymaxe +2, xmine-1:xmaxe + 2]
    return im, ch, coul



##on identifie le motif: d'abord la couleur
def compte_pix_zone(motif):
    """on compte le nombre de pixels rouge, noir, ou blanc présents dans le motif"""
    rnb = [0,0,0]
    for i in range(len(motif)):
        for j in range (len(motif[0])):
            if motif[i][j][0] == 0:
                rnb[1] = rnb[1] + 1
            elif motif[i][j][1] == 1:
                rnb[2] = rnb[2] + 1
            else:
                rnb[0] = rnb[0] + 1
    return rnb

def conv_vert(coul):
    """permet de différencier les trèfles des piques selon une 'convexité verticale',
    i.e si sur une ligne verticale, dans la partie supérieure du motif, on a une alternance couleur blanc couleur"""
    etat = 0
    for x in range (len(coul[0])):
        cpt = 0
        for y in range(3*len(coul)//5):
            if (cpt == 0 or cpt == 2) and coul[y][x][1] != 0:
                cpt += 1
            elif cpt == 1 and coul[y][x][1] == 0:
                cpt+= 1
        if cpt >= 3 :
            etat += 1
    return etat < 3

def conv_hor(coul):
    """permet de différencier les coeurs des carreaux selon une 'convexité horizontale',
    i.e si sur une ligne horizontale, dans la partie supérieure du motif, on a une alternance couleur blanc couleur"""
    etat = 0
    for y in range (1, 6):
        cpt = 0
        for x in range(len(coul[0])):
            if cpt%2 == 0 and coul[y][x][1] == 0:
                cpt += 1
            elif cpt%2 == 1 and coul[y][x][1] != 0:
                cpt+= 1
        if cpt >= 3:
            etat += 1
    return etat < 3

def find_couleur(coul):
    """arbre de décision pour trouver quelle est la couleur"""
    [rsup, nsup, bsup] = compte_pix_zone(coul[:len(coul)//2])
    [rinf, ninf, binf] = compte_pix_zone(coul[len(coul)//2:])
    if rsup + rinf > nsup + ninf: #couleur rouge
        if conv_hor(coul):
            return "carreau"
        return "coeur"
    if conv_vert(coul):
        return "pique"
    return "trefle"

##puis le chiffre
def is_white(img, x, y):
    """teste si le pixel (x,y) est blanc"""
    return img[y][x][0] == 1 and img[y][x][1] ==  1 and img[y][x][2] == 1

def dfs(img, vus, pixel, number):
    """parcours en profondeur d'une composante connexe blanche d'une image en marquant tous les points visités"""
    pile = []
    pile.append(pixel)
    while pile != []:
        (x, y) = pile.pop()
        vus[y][x] = number
        if is_white(img, x, y):
            if x < len(img[0])-1 and vus[y][x+1] == 0:
                pile.append((x+1, y))
            if x > 0 and vus[y][x-1] == 0:
                pile.append((x-1, y))
            if y > 0 and vus[y-1][x] == 0:
                pile.append((x, y-1))
            if y < len(img)-1 and vus[y+1][x] == 0:
                pile.append((x, y+1))

def find_pixel(img, vus):
    """on recherche une composante blanche connexe pas encore explorée et on renvoie les coordonnées d'un de ses pixels"""
    for i in range(len(img)):
        for j in range(len(img[0])):
            if vus[i][j] == 0 and is_white(img, j, i) and ((i<len(img[0]) -1 and is_white(img, j, i+1)) or (j<len(img)-1 and is_white(img, j+1, i)) or (j>0 and is_white(img,j-1,i)) or (i>0 and is_white(img,j,i-1))):
                return (i,j)
    return (-1, -1)

def n_comp_connexes(img, vus):
    """compte le nombre de composantes connexes de l'arrière plan (1, 2 ou 3) en vérifiant que chaque composante compte + d'1 pixel sinon elle est ignorée"""
    #parcours en profondeur de la zone blanche pendant lequel on marque tous les points qu'on visite
    dfs(img, vus, (0,0), 1)
    #on explore de nouveau l'image : on cherche s'il y a une autre composante connexe et on repère les coordonnées d'un de ses pixels
    (i, j) = find_pixel(img, vus)
    if i == -1:
        return 1
    else:
        dfs(img, vus, (j,i), 2)
        (i, j) = find_pixel(img, vus)
        if i==-1:
            return 2
        else:
            return 3

def ligne_vert(chiffre, ratio = -1):
    """détecte la présence d'une ligne verticale dans le chiffre, de largeur >= 5 pixels et de longueur >= taille motif -5 pixels et renvoie les coordonnées du pixel où débute la ligne"""
    k = 5
    if ratio > 0:
        k = len(chiffre[0])//ratio
    for x in range(len(chiffre[0])-k):
        [r, n, b] = compte_pix_zone(chiffre[1:len(chiffre) -1, x:x+k])
        if b <= 20:
            return x
    return -1

def ligne_hor(chiffre):
    """de même que pour la ligne verticale, mais pour une ligne horizontale"""
    for y in range(len(chiffre)-5):
        [r,n,b] = compte_pix_zone(chiffre[y:y+5, 1:len(chiffre[0])-1])
        if b<=15:
            return y
    return -1

def taille_comp(num, vus):
    """compte le nombre de pixels blancs dans la seconde composante connexe (la boucle)"""
    n = 0
    for i in range(len(vus)):
        for j in range(len(vus[0])):
            if vus[i][j] == 2 and is_white(num, j, i):
                n = n+1
    return n

def is_ten(num):
    """on repère la présence d'un bout du 1 sur le bord de l'image"""
    for i in range(len(num)-1, -1, -1):
        if num[i][0][0] != 1:
            return True
    return False

def symetrie_vert(num):
    """la figure est-elle symétrique (à peu près) selon un axe de symétrie horizontale?"""
    k = len(num)//2
    [r, n, b] = compte_pix_zone(num[k:])
    diff = 0#calcul de la différence symétrique
    for i in range(k):
        for j in range(len(num[0])):
            if is_white(num, j, i) != is_white(num, j, i+k):
                diff += 1
    return diff/(r+n)
                

def desequilibre(num, sup=True):
    """on recherche un déséquilibre de l'image autour de l'axe vertical traversant le plus haut pixel (si sup) ou le plus bas (sinon) de l'image (en cas de multiplicité, on prend celui le plus à gauche)
    On ne renvoie pas dans quel sens se produit le deséquilibre (problèmes de symétrie), simplement sa présence ou non"""
    i = 0
    j = 0
    cherche = True
    while cherche and (i<len(num) or j<len(num[0])):
        if sup and num[i][j][1] != 1 or (not sup and num[len(num) - 1 - i][j][1] !=1):
            cherche = False
        elif j == len(num[0]) - 1:#fin de la ligne
            i = i+1
            j = 0
        else:
            j = j+1
    [rg, ng, bg] = compte_pix_zone(num[:, :j])
    [rd, nd, bd] = compte_pix_zone(num[:, j:])
    return abs((rg + ng - rd - nd)) >= (rg+ng+bg+rd+nd+bd)//10

def add_line(num):
    """pour différencier le 3 du 5 on ajoute une ligne verticale puis on appelle n_comp_connexes: si il y en a 2, alors c'est un 5, sinon c'est un 3
    renvoie si le motif est un 5"""
    new_img = deepcopy(num)
    vus = np.zeros((len(num), len(num[0])))
    i = 0
    j = 0
    cherche = True
    while cherche and (i<len(num) or j<len(num[0])):
        if num[i][j][1] != 1:
            cherche = False
        elif j == len(num[0]) - 1:#fin de la ligne
            i = i+1
            j = 0
        else:
            j = j+1
    new_img[1:len(num) - 1,j:j+5] = [0,0,0]
    n = n_comp_connexes(new_img, vus)
    return n == 2
    

def find_number(num):
    """arbre de décision afin d'identifier le chiffre de la carte"""
    vus = np.zeros((len(num), len(num[0])))
    n = n_comp_connexes(num, vus)
    if n==1: #2, 3, 5, 7, 10 ou V -> le 10 en fait partie si c'est le 1 qui est identifié
        if symetrie_vert(num) <= 1/6 : #10 ou 3
            if ligne_vert(num, 4) == -1:
                return "3"
            return "10"
        y = ligne_hor(num)
        if y == -1:#3, 5 ou V: attention le 5 n'a pas de barre horizontale car elle ne couvre pas l'intégralité de la largeur
            if desequilibre(num, False):#3 ou 5
                if add_line(num):
                    return "5"
                return "3"
            return "V"
        elif y <= len(num)//2:
                return "7"
        return "2"           
    elif n==2: #A, 4, 6, 9, 10, D ou R
        x = ligne_vert(num)
        if x==-1:#A, 6, 9 ou 10
            [rsup, nsup, bsup] = compte_pix_zone(num[:len(num)//2])
            [rinf, ninf, binf] = compte_pix_zone(num[len(num)//2:])
            if rsup+nsup - (rinf+ninf) > (rsup+nsup+rinf+ninf)//8:
                return "9"
            elif abs(rsup+nsup - (rinf+ninf)) <= (rsup+nsup+rinf+ninf)//8:
                return "10"
            else:
                if desequilibre(num):
                    return "6"
                else:
                    return "1"
        elif x < len(num[0])//2:#D ou R
            if taille_comp(num, vus) >= 700:
                return "D"
            else:
                return "R"
        else:
            return "4"
    else:
        return "8"
        

    

##fonctions d'affichage des droites et des coins
def display_line(lg, h, x):
    """affiche une ligne lg définie par le tuple (a, b, c) telle que a*x + b*y + c = 0"""
    a, b, c = lg
    if b == 0: #droite verticale
        plt.plot([-int(c/a), -int(c/a)], [0, h], 'r')
    else:
        plt.plot(x, -(a/b)*x - c/b, 'r')

def display_corners(cn):
    """affiche une liste de coins"""
    assert (len(cn) == 4), "Pas assez de coins détectés"
    absc = [cn[0][0], cn[1][0], cn[3][0], cn[2][0]]
    ordo = [cn[0][1], cn[1][1], cn[3][1], cn[2][1]]
    plt.plot(absc, ordo, 'o')

def bords_col(skel, h, l):
    """renvoie l'image binaire (en r, g, b) associée au contour de la carte"""
    m = np.zeros((h,l,3))
    for (x, y) in skel:
        m[y][x] = [1, 1, 1]
    return m

def bords_g(skel, h, l):
    """renvoie l'image binaire (en nuances de gris) associée au contour de la carte"""
    m = np.zeros((h,l))
    for (x, y) in skel:
        m[y][x] = 1
    return m

###     --------------
###     CODE PRINCIPAL
###     --------------


def analyse_carte(nom_carte, chemin = "photo_cartes/", m_temps = False, affichage = False):
    """
    Entrées :
        nom_carte : chaîne de caractères telle que la photo de la carte qu'on cherche à identifier est située à l'adresse chemin + nom_carte + ".jpg"
        chemin : chemin relatif pour trouver la carte
        m_temps : indique si on veut le détail de temps des opérations. Surtout utile pour des statistiques
        affichage : indique si on veut l'affichage du détail des opérations
    Sortie :
        Par défaut :
            tuple contenant, en chaîne de caractères, la valeur et l'enseigne de la carte
        Si m_temps :
            tuple contenant (chacun en chaîne de caractères):
                * la valeur
                * l'enseigne
                * le temps du filtre de Canny (en s)
                * le temps de la transformation de Hough (en s)
                * le temps de reconstitution du coin (en s)
                * le temps d'identification de la couleur (en s)
                * le temps d'identification de la valeur (en s)

    Résultat :
        Détermine la valeur et l'enseigne de la carte prise en photo. Si affichage, affiche le déroulement de l'algorithme.

    Attention :
        Conditions nécessaires au fonctionnement de l'algorithme :
            * 1 seule carte sure l'image
            * très bon contraste, et bonne qualité de l'image (si possible fond noir)
            * pas de flou
            * pas d'occlusion de la carte

        L'algorithme a presque exclusivement été testé sur des cartes de 3120*4160 pixels.
        Pour des résolutions plus faibles, il risque d'être moins fiable, et pour des plus importantes, de prendre un temps déraisonnable.

        L'algorithme est supposé être assez robuste à des perspectives et rotations importantes, si ces conditions sont respectées.
    """
    img_pil = Image.open(chemin+nom_carte+".jpg")
    width, height = img_pil.size
    img_compr = img_pil.resize((width//15, height//15)) #ce paramètre peut éventuellement être modifié si on compte travailler avec des images d'autres dimensions
    img_compr = np.array(img_compr)
    img_compr = ndimage.gaussian_filter(img_compr, sigma = 1.3)

    img = rgb2gray(img_compr)
    img_fs = np.array(img_pil)
    grad = gradient(img)

    #filtre de Canny
    deb_bord = time()
    sobel = canny(grad)
    t_canny = str(time() - deb_bord)

    ##A) ON LOCALISE LA CARTE

    #transformation de Hough
    deb = time()
    edges = thinning(sobel, contour(sobel))
    img_link = bords_g(edges, len(sobel), len(sobel[0]))
    
    clusters = dfs_clusters(img_link)
    img1 = bords_col(edges, len(sobel), len(sobel[0]))
    lignes = hough_amelioree(clusters, len(sobel), len(sobel[0]))

    #calcul des coins
    l_coins = coins(lignes, img)
    t_hough = str(time() - deb)

    ##B) ON REDRESSE LA CARTE ET DÉTECTE LE SYMBOLE ANALYSABLE
    #on repasse à l'image non compressée
    deb = time()
    im, ch, coul = reconstitue_rectangle(l_coins, 100, img_fs) 
    t_sym = str(time() - deb)
    
    deb = time()
    sc = find_number(ch)
    t_ch = str(time() - deb)

    deb = time()
    nom_c = find_couleur(coul)
    t_col = str(time() - deb)

    ##C) AFFICHAGE
    if affichage:
        plt.figure()
        plt.gray()
        plt.subplot(2,3,1)
        plt.title("image en noir et blanc")
        plt.axis('off')
        plt.imshow(img)
        plt.subplot(2,3,2)
        plt.title("détection des contours")
        plt.axis('off')
        plt.imshow(sobel)
        plt.subplot(2,3,3)
        plt.axis('off')
        plt.imshow(img1)
        plt.xlim(0, len(sobel[0])-1)
        plt.ylim(len(sobel)-1, 0)
        x = np.arange(0, len(sobel[0]))
        for lg in lignes:
            display_line(lg, len(sobel), x)
        display_corners(l_coins)
        plt.subplot(2,3,4)
        plt.title("symbole")
        plt.axis('off')
        plt.imshow(im)
        plt.subplot(2,3,6)
        plt.axis('off')
        plt.title(f"chiffre: {sc}")
        plt.imshow(ch)
        plt.subplot(2,3,5)
        plt.axis('off')
        plt.title(f"couleur: {nom_c}")
        plt.imshow(coul)
        plt.show()

    if m_temps:
        return (sc, nom_c, t_canny, t_hough, t_sym, t_col, t_ch)
    return (sc, nom_c)
        
