import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io 
import matplotlib.tri as mtri
import matplotlib.tri.triangulation as tri 
from mpl_toolkits.mplot3d import Axes3D



#%% PARAMÈTRES DÉTERMINANTS LE CALCUL
################################################

maille = 1  # Choix du maillage 1 (gros),2 (plus fin) ,3 (encore plus fin) ou 4 (trou)

# Proprietes mecaniques
E = 1000 # Young's modulus  
v = 0.3  # Poisson's coefficient

# Forces (densité surfacique ici)
fvx = 100          
fvy = 0 
fv = np.array([fvx,fvy])



#%% LECTURE ET CHARGEMENT DU MAILAGE
#######################################################

if maille == 1:
    mat = scipy.io.loadmat('./maillage_1.mat')
elif maille ==2:
    mat = scipy.io.loadmat('./maillage_2.mat')
elif maille ==3:
    mat = scipy.io.loadmat('./maillage_3.mat')
elif maille ==4:
    mat = scipy.io.loadmat('./maillage_trou.mat')
else:
    raise ValueError('Maille 1,2,3 ou 4 ')
    
# paramètres déduits
#===============================================

# coordonnes des noeuds  
X = mat['p'].T  

# table de connectivite
C = mat['t'].T[:,:-1] -1   

# nombre de noeuds du maillage
n_nodes = X.shape[0] 

# nombre d'elements du maillage
n_elems = C.shape[0] 

# Nombre de noeuds par element
n_nodes_elem = 3 # ce sont des triangles P1

# Nombre de ddl par element
ndofe= 2*n_nodes_elem # champ vectoriel en 2D

# Nombre de ddl dans tout le domaine  
ndof = 2*n_nodes # champ vectoriel en 2D


# dimensions de la plaque 
xmin =  np.min(X[:,0])
xmax =  np.max(X[:,0])
ymin =  np.min(X[:,1])
ymax =  np.max(X[:,1])
# vérifier dimensions de la plaque [0,2]x[0,1]

# affichage du maillage
Initial_triangulation =  tri.Triangulation(X[:,0],X[:,1],C)
plt.figure(0)
plt.triplot(Initial_triangulation, color = 'black')
plt.title("maillage")
plt.show()


#%% CALCUL DES FUNS EF P1 LAGRANGE
############################################
 
def fun_tri_P1_lag(x,y,x_nodes,y_nodes):
    # x,y -> point d'évaluation
    # x_nodes -> tableau 1D numpy avec abscisses des noeuds (ceux qui faut)
    # y_nodes -> tableau 1D numpy avec ordonnées des noeuds (ceux qui faut)
    #calcoul dou Te
    Te=0.5*np.abs((x_nodes[1]-x_nodes[0])*(y_nodes[2]-y_nodes[0])-(y_nodes[1]-y_nodes[0])*(x_nodes[2]-x_nodes[0]))

    N1=(x_nodes[1]*y_nodes[2]-x_nodes[2]*y_nodes[1])+(y_nodes[1]-y_nodes[2])*x+(x_nodes[2]-x_nodes[1])*y
    N2=(x_nodes[2]*y_nodes[0]-x_nodes[0]*y_nodes[2])+(y_nodes[2]-y_nodes[0])*x+(x_nodes[0]-x_nodes[2])*y
    N3=(x_nodes[0]*y_nodes[1]-x_nodes[1]*y_nodes[0])+(y_nodes[0]-y_nodes[1])*x+(x_nodes[1]-x_nodes[0])*y
    
    N = 1/(2*Te)*np.array((N1,N2,N3))
    
    # calcoul da dérivé en x
    dxN1=(y_nodes[1]-y_nodes[2])
    dxN2=(y_nodes[2]-y_nodes[0])
    dxN3=(y_nodes[0]-y_nodes[1])
    
    dNdx = 1/(2*Te)*np.array((dxN1,dxN2,dxN3))
    
    #calcoul da dérivée en y
    dyN1=(x_nodes[2]-x_nodes[1])
    dyN2=(x_nodes[0]-x_nodes[2])
    dyN3=(x_nodes[1]-x_nodes[0])
    
    dNdy = 1/(2*Te)*np.array((dyN1,dyN2,dyN3))
    
    return [N,dNdx,dNdy]
    # N = [N1(x,y),N2(..),N3(..)]-> tableau 1D numpy avec valeur des 3 funs au point d'éval
    # dNdx -> idem avec dérivée par rapport à x des funs
    # dNdy -> idem avec dérivée par rapport à y des funs
    
def GetBe(dNdx,dNdy):
    # dNdx=[dN1dx(x,y),dN2dx(x,y),dN3dx(x,y)] -> tableau 1D numpy avec valeur à un certain point d'éval
    
    Be = np.zeros((n_nodes_elem,ndofe))
    Be[0,0:3]=dNdx
    Be[1,3:]=dNdy
    Be[2,0:3]=dNdy
    Be[2,3:]=dNdx
    
    return Be
    # Be : matrice élémentaire lien déformation - déplacement (evaluee à un certain point)


def GetNe(N):
    
    Ne_matrix=np.zeros((n_nodes_elem-1,ndofe))
    Ne_matrix[0,0:3]=N
    Ne_matrix[1,3:]=N
    
    return Ne_matrix
    # Ne_matrix : matrice élémentaire lien déplacement - ddls 

#%%
#--------------------------------------------------------------------------
#
# 1. CALCUL DES DEPLACEMENTS
#
#--------------------------------------------------------------------------
    

#%% CALCUL DE LA MATRICE DE RIGIDITE
#######################################################
    

# Loi de comportement : Contraintes - Deformations
# Hypothese de contraintes planes 
# convention : [sigmaxx,sigmayy,sigmaxy] = H [epsxx, epsyy, 2*epsxy] 
H = np.array([[E/(1-v**2),v*E/(1-v**2),0],
              [v*E/(1-v**2),E/(1-v**2),0],
             [0,0,E/(2*(1+v))]])

    
# initialisation de la matrice de rigiditié 
K = np.zeros((ndof,ndof)) 

# Boucle sur les éléments

for e in range(n_elems):
    
    # coordonnes des noeuds de l'element e   
    numerou = C[e,:]
    x1,y1 = X[numerou[0],:]
    x2,y2 = X[numerou[1],:]
    x3,y3 = X[numerou[2],:]
    
    x_nodes = np.array([x1,x2,x3])
    y_nodes = np.array([y1,y2,y3])
    
    x,y = (0,0)
    
    # calcul de la matrice de rigidité elementaire
    N,dNdx,dNdy = fun_tri_P1_lag(x, y, x_nodes, y_nodes)
    
    Be = GetBe(dNdx, dNdy)
    
    Te=0.5*np.abs((x_nodes[1]-x_nodes[0])*(y_nodes[2]-y_nodes[0])-(y_nodes[1]-y_nodes[0])*(x_nodes[2]-x_nodes[0]))
    Ke = Te*(Be.T@H@Be)
     
    # Assemblage des contributions élémentaires
    for i in range(n_nodes_elem):
        for j in range(n_nodes_elem):
            K[C[e,i],C[e,j]] += Ke[i,j]
            K[C[e,i]+n_nodes,C[e,j]] += Ke[i+n_nodes_elem,j]
            K[C[e,i],C[e,j]+n_nodes] += Ke[i,j+n_nodes_elem]
            K[C[e,i]+n_nodes,C[e,j]+n_nodes] += Ke[i+n_nodes_elem,j+n_nodes_elem]
            

#%% CALCUL DU SECOND MEMBRE
#######################################################
           
# initialisation du second membre
F = np.zeros(ndof)

for e in range(n_elems):
    numerou = C[e,:]

    x1, y1 = X[numerou[0],:]
    x2, y2 = X[numerou[1],:]
    x3, y3 = X[numerou[2],:]
    
    x_nodes = np.array([x1, x2, x3])
    y_nodes = np.array([y1, y2, y3])

    Te = 0.5 * np.abs((x_nodes[1]-x_nodes[0])*(y_nodes[2]-y_nodes[0])- (y_nodes[1]-y_nodes[0])*(x_nodes[2]-x_nodes[0]))
    N1,dNdx,Dndy =fun_tri_P1_lag(x1, y1, x_nodes, y_nodes)
    N2,dNdx,Dndy =fun_tri_P1_lag(x2, y2, x_nodes, y_nodes)
    N3,dNdx,Dndy =fun_tri_P1_lag(x3, y3, x_nodes, y_nodes)
    Fe = Te/3 *(GetNe(N1).T @ fv+GetNe(N2).T @ fv+GetNe(N3).T @ fv)


    for i in range(len(numerou)):
        F[numerou[i]] += Fe[i]
        F[numerou[i]+n_nodes] += Fe[i+n_nodes_elem]
print(F)

#%% IMPOSITION DES CL DE DIRICHLET
#######################################################
for n in range(n_nodes):

    x, y = X[n,:]

    if x< 1e-10:   
        i = n
        K[i,:] = 0
        K[:,i] = 0
        K[i,i] = 1
        F[i] = 0


        i = n + n_nodes
        K[i,:] = 0
        K[:,i] = 0
        K[i,i] = 1
        F[i] = 0

    
#%% RESOLUTION DU SYSTÈME LINÉAIRE ET VISUALISATION
#######################################################
        
# résolution
U = np.linalg.solve(K,F)

#Calcul des coordonnes des noeuds apres deformation
x = X[:,0] + U[:n_nodes]
y = X[:,1] + U[n_nodes:]

# affichage du maillage
Deformed_triangulation  =  tri.Triangulation(x,y,C)
plt.figure(1)
plt.triplot(Initial_triangulation, color = 'black')
plt.triplot(Deformed_triangulation, color = 'blue')
plt.show()
 

#%%
#--------------------------------------------------------------------------
#
# 2. POST-TRAITEMENT : CALCUL DES CONTRAINTES DE VON MISES
#
#--------------------------------------------------------------------------
    

# initialisations

T = np.zeros(n_nodes)       # Surface
SVM = np.zeros(n_nodes)     # contrainte de Von Mises


#boucle sur les elements 

for e in range(n_elems):
    
    # coordonnes des noeuds de l'element e   
    x_nodes = X[C[e,:],0]
    y_nodes = X[C[e,:],1] 
    
    
    x1=x_nodes[0]; x2=x_nodes[1]; x3=x_nodes[2] 
    y1=y_nodes[0]; y2=y_nodes[1]; y3=y_nodes[2] 
    
    # surface de l'élément e
    Te = 0.5*np.abs((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1))
    
    # Calcul des contraintes de Von Mises
    [Nm,dNmdx,dNmdy] = fun_tri_P1_lag(x1,y1,x_nodes,y_nodes)
    Be = GetBe(dNmdx,dNmdy)
    Ue = np.r_[U[C[e,:]],U[C[e,:]+n_nodes]]
    sigma = H.dot(Be.dot(Ue))
    sxx = sigma[0] 
    syy = sigma[1]  
    sxy = sigma[2] 
    svm = (sxx**2+syy**2+3*sxy**2-sxx*syy)**0.5 
    
    # Assemblage : surface et sigma
    for i in range(n_nodes_elem):
        T[C[e,i]] +=  Te
        SVM[C[e,i]] +=  Te*svm

# Calcul de la contrainte de Von Mises aux noeuds
SVM = SVM/T

# Visualisation des containtes 
t = mtri.Triangulation(x,y,C)
plt.figure(2)
plt.triplot(t)
plt.tricontourf(t,SVM)
plt.colorbar()
plt.title('Contraintes de Von Mises')
 
 
    



 
