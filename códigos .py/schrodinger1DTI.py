import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from numba import njit

dx=0.1 #Define dx (angstroms)
RY=13.60569 #E*RY: Rydberg -> ev
ang=1.88973 #X/ang: raio de bohr -> angstroms

dx=dx*ang #dx de angstrom para raio de bohr
a=-1/(dx**2) #Define 'a' por conveniência

x=np.arange(-100*ang ,100*ang,dx) #Define o vetor posição (raio de bohr)
N, L=(x.size, x.max()-x.min()) #Armazena o tamanho do vetor posição e o comprimento do espaço que ele representa

V=sparse.diags([0],[0],shape=(N,N)).toarray() #Cria uma matriz NxN com o potencial V(x)
H=sparse.diags([a,-2*a,a],[-1,0,1],shape=(N,N)).toarray() #Cria a matriz Hamiltoniano (apenas com a parte cinética)
H=H+V #soma a matriz do potencial V(x)
eig, psi=np.linalg.eigh(H) #Resolve o sistema e atribui-se: autovalores->eig, autovetores->psi
prob=np.abs(psi)**2 #Atribui a densidade de probabilidade |Psi|² a variável prob

V=np.diag(V,k=0) #transforma a matriz potencial em um vetor 1D para plotar

eig=eig*RY #Ryd->eV
x=x/ang #RBohr->angstroms

nmin, nmax=(0,10)
@njit
def norm(prob,nmin,nmax,En):
    mmax=np.zeros(nmax+2)
    for i in range(0,nmax+2):
        mmax[i]=(En[i+1]-En[i])*0.5
    for i in range(0,nmax+2):
        prob[:,i]=prob[:,i]*np.abs(mmax[i]/prob[:,i].max())
    return prob

fig, ax=plt.subplots(figsize=(16,12))
ax.set_xlabel('$x\, (\AA)$')
ax.set_ylabel('Energia $(eV)$')
ax.set_xlim([x.min()-50, x.max()+ 50])
plt.title('Autoenergias e as densidades de probabilidades correspondentes')
prob=norm(prob,nmin,nmax,eig)

for i in range(nmax,nmin-1,-1):
    fig=plt.plot(x,prob[:,i]+eig[i],label=f'$E_{i}={eig[i]:2f}$ eV')
    
fig=plt.plot(x,V,'k',label='$V(x)$')
leg=plt.legend(loc='upper right')
plt.show()