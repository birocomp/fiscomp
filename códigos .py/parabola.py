#Esse código resolve a 2a lei de Newton para um lançamento parabólico
#com uma força de arrasto proporcional a v²
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

def main():
    t, dt, x, y, vx, vy, ax, ay, N, g, b = var()
    x, y, vx, vy, Fi, ax, ay = prop(x,y,vx,vy,ax,ay,dt,N,b,g)
    x.resize(Fi)
    y.resize(Fi)
    vx.resize(Fi)
    vy.resize(Fi)
    ay.resize(Fi)
    ax.resize(Fi)
    t=np.resize(t,Fi)
    VGA=[True, True, True]
    plot(x,y,vx,vy,ax,ay,Fi,g,7,VGA,dt,t)
    plotvst(x,y,vx,vy,ax,ay,t,dt)
    
def var():
    N, g, b, m = (1000, 9.798, 1, 10) #N é o numero de pontos máximos, g a gravidade, b constante da força de arrasto e m a massa
    b=b/m
    t, dt = np.linspace(0, 4, N, retstep=True)
    x, y = np.zeros(N), np.zeros(N)
    vx, vy = np.zeros(N), np.zeros(N)
    ax, ay = np.zeros(N), np.zeros(N)
    vx[0], vy[0] = (1., 1.)
    v = np.sqrt(vx[0]**2 + vy[0]**2)
    x[0], y[0] = (0., 0.)
    ax[0], ay[0] = (-b*v*vx[0], -b*v*vy[0])
    return t, dt, x, y, vx, vy, ax, ay, N, g, b
    
def plot(x,y,vx,vy,ax,ay,Fi,g,npontos,VGA,dt,t):
    imax=x.size//npontos + 1
    fig, axes=plt.subplots(figsize=(16,9))
    fig=plt.plot(x,y,'k--')
    fig=plt.hlines(0,x.min(),x.max(),color='k',ls='--')
    ximax=x.size
    fig=plt.plot(x[::imax],y[::imax],'ko')
    if VGA[0]==True:
        fig=plt.quiver(x[::imax],y[::imax],vx[::imax],vy[::imax],
        color='red',units='xy', width=.0005, label='Velocidade')
    if VGA[1]==True:
        fig=plt.quiver(x[::imax],y[::imax],ax[::imax],ay[::imax],
        color='blue',units='xy', width=.0005, label='Força de Arrasto')
    if VGA[2]==True:
        fig=plt.quiver(x[::imax],y[::imax],0,-g,
        color='green',units='xy', width=.0005, label='Gravidade')
    axes.axis('equal')
    plt.title(f'Trajetória | Alcance máximo = {x[ximax-1]: .2f} metros | $\Delta t=${dt:.2e} s')
    plt.legend()
    
def plotvst(x,y,vx,vy,ax,ay,t,dt):
    fig1, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True,figsize=(16,9))
    ax1.plot(t,y,'o-',label='Altura: $y$',color='k')
    ax1.legend()
    ax2.plot(t,np.sqrt(vx**2+vy**2),'o-',label='Velocidade: $v = \sqrt{v_x^2 + v_y^2}$',color='red')
    ax2.legend()
    ax3.plot(t,np.sqrt(ax**2+ay**2),'o-',label='Força de Arrasto: $-bv^2$',color='blue')
    ax3.legend()
    
@njit
def prop(x,y,vx,vy,ax,ay,dt,N,b,g):
    def xx(x,vx,dt): #estima x(t+dt)
        return x+vx*dt
    
    def yy(y,vy,dt): #estima y(t+dt)
        return y+vy*dt
    
    def v_x(vx,ax,dt): #estima vx(t+dt)
        return vx+ax*dt
    
    def v_y(vy,ay,dt,g): #estima vy(t+dt)
        return vy+(ay-g)*dt
    
    def a_x(b,vx,v): #define ax(t+dt)
        return -b*v*vx
    
    def a_y(b,vy,v): #define ay(t+dt)
        return -b*v*vy
    
    for i in range(0,N-1,1):
        if y[i]<0: #para o loop se y<0
            Fi=i+1
            break
        vx[i+1]=v_x(vx[i],ax[i],dt)
        vy[i+1]=v_y(vy[i],ay[i],dt,g)
        x[i+1]=xx(x[i],vx[i],dt)
        y[i+1]=yy(y[i],vy[i],dt)
        v=np.sqrt(vx[i]**2 + vy[i]**2)
        ax[i+1]=a_x(b,vx[i+1],v)
        ay[i+1]=a_y(b,vy[i+1],v)
    return x, y, vx, vy, Fi, ax, ay
    
main()