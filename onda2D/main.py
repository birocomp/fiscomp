import matplotlib.pyplot as plt
from numba import jit, cuda
import numpy as np
import matplotlib.animation as animation
import gvars,pot

#import variables from 'gvars.py' as globals
#any additional variable in 'gvars.py' must be received here
#in the same order
nx,ny,dt,dx,nmax,nf,v,e,dx2,nt,x,y,X,Y,fr,t,tfr,V,u,ut,ut1,U=gvars.init()

def main():
	global U,nt,V,u,ut
	
	V=pot.oneslit(C=(-200.0,0),WW=15.0,SH=150.0,horizontal=False)
	u=u0(u)
	ut=ut0(ut)
	
	timeprop(u,pmax=10)
	
	U=np.real(U)**2
	
	plot(U,V,ov_UV=True,alpha=0.5,autolim=True,
					pngV=True,gif=False,mp4=True,
					Ucmap='viridis',Vcmap='gray',fps=15,sx=10,sy=10,dpi=60)
	
@jit
def u0(u):
	#u[:,:]=np.real(np.exp(-0.1*((X[:,:]+400.0)**2+(Y[:,:])**2)+1.j*(0.1*X[:,:])))
	return u
	
@jit
def ut0(ut):
	#ut[:,:]=np.real(-1.j*v*np.exp(-0.1*((X[:,:]+400.0)**2+(Y[:,:])**2)+1.j*(0.1*X[:,:])))
	return ut
	

def timeprop(u,pmax):
	global ut,nt,U

	print("Time evolution started...")
	if pmax!=0:
		if nmax%pmax==0:
			pnmax=np.int(nmax/pmax)
		else:
			pnmax=nmax
	for n in range(0,nmax):	#Time loop from t=0 to t=(nmax-1)*dt'
		u=u*V
		u[195:205,0:5]=np.exp(-(t[n]-0.0)**2/(50*dt)**2)
		if nt+1<=nf and tfr[nt]==t[n]: #check if u should be saved
			U[1:nx,1:ny, nt]=u[1:nx,1:ny]			
			nt=nt+1

		evol(ut1,ut,u)#evolves u
		ut=ut1
		
		if pmax!=0:
			if (n%pnmax)==0: #check if progress is to be displayed
				print('Calculation: '+str("{:.1f}".\
									format((100.0*n)/nmax))+' %')						
	
	print("Time evolution ended")
	
def plot(U,V,autolim,ov_UV,alpha,pngV,gif,mp4,Ucmap,
											Vcmap,fps,sx,sy,dpi):	
	
	fig, ax=plt.subplots(figsize=(sx,sy))
	ex=x.min()+dx, x.max()-dx, y.min()+dx, y.max()-dx
	im1 = plt.imshow(U[1:nx,1:ny,0], cmap=Ucmap, extent=ex, 
										vmin=np.min(U), vmax=np.max(U))
	cb = fig.colorbar(im1, ax=ax)
	
	alphaV=V*0
	
	for i in range(1,nx+1):
		for j in range(1,ny+1):
			if V[i,j]==0:
				alphaV[i,j]=alpha
			else:
				alphaV[i,j]=0
	
	if ov_UV==True:
		im2 = plt.imshow(V[1:nx,1:ny], cmap=Vcmap, alpha=alphaV, extent=ex)

	wav = animation.FuncAnimation(fig, animate, 
								fargs=(im1, autolim), frames=nt)
	
	if mp4==True:
		print("Rendering MP4 frames...")
		FFwriter = animation.FFMpegWriter(fps=fps)
		wav.save('biwav.mp4', writer = FFwriter, dpi=dpi)
		print("MP4 saved")
	
	if gif==True:
		print("Rendering GIF frames...")
		wav.save('2Dwav.gif', writer = 'imagemagick', dpi=dpi, fps=fps)
		print("GIF saved")
	
	if pngV==True:
		imV, ax3 = plt.subplots(figsize=(sx,sy))
		im3 = plt.imshow(V[1:nx,1:ny], cmap=Vcmap, extent=ex)
		cb1 = fig.colorbar(im3, ax=ax3)
		imV = plt.gcf()		
		imV.savefig('2Dpot.png', dpi=dpi)
		
def animate(i,im1,autolim):#render frame i+1 from frame 1 to nf	
	Un=U[1:nx,1:ny,i]
	umax=np.max(np.abs(Un))
	if autolim==True:
		im1.set_array(Un/umax)
		im1.set_clim(0,1)
	else:
		im1.set_array(Un)

def writeinfo():#write info file about simulation parameters
	pass
	xmax,ymax=np.amax(x),np.amax(y)
	xmin,ymin=np.amin(x),np.amin(y)
	tmin,tmax=np.amin(t),np.amax(t)
	Ftmin,Ftmax=np.amin(tfr),np.amax(tfr)
	f.write("SPACE RELATED INFO\n")
	f.write("(xmin,xmax)= %f,%f;\n(ymin,ymax)= %f,%f\n"\
	%(xmin,xmax,ymin,ymax))
	f.write("(LX,LY)=(%d,%d) \n" %(xmax-xmin,ymax-ymin))
	f.write("(nx,ny)=(%d,%d); dx= %.5f \n \n" %(nx,ny,dx))
	f.write("TIME RELATED INFO \n")
	f.write("nmax= %d; frames_saved=%d; dt=%f \n" %(nmax,nt,dt))
	f.write("(tmin,tmax)=(%f,%f)\n" %(tmin,tmax))
	f.write("(Ftmin,Ftmax)=(%f,%f)\n \n" %(Ftmin,Ftmax))
	f.write("SYSTEM RELATED INFO\n")
	f.write("m=%f; h=%f\n" %(m,h))
	f.write("kx=%f; ky=%f; sigma=%f\n" %(kx,ky,sigma))
	f.write("X0=%f; Y0=%f\n" %(x0,y0))
	f.write("U0=%f; K0=%f; E0=%f" %(um,km,um+km))
	f.close()

@jit
def trapz(p,xi,xf,yi,yf):#double integrates 'p' using trapezoidal rule 
	psum=(p[xi,yi]+p[xf,yf]+p[xi,yf]+p[xf,yi])*(xf-xi)*(yf-yi)*0.25
	return psum21

@jit
def dif(p):	#Calculates  2D laplacian of function 'p'
	dp=0*p
	for i in range(1,nx+1):
		for w in range(1,ny+1):
			dp[i,w]=(p[i+1,w]+p[i,w+1]-4.0*p[i,w]+p[i-1,w]+p[i,w-1])*dx2
	return dp

@jit
def evol(ut1,ut,u):	#Evolves u(t) to u(t+dt)
	uxx=dif(u)
	for i in range(1,nx+1):
		for w in range(1,ny+1):
			ut1[i,w]=v*v*uxx[i,w]+ut[i,w]
	for i in range(1,nx+1):
		for w in range(1,ny+1):
			u[i,w]=ut[i,w]*dt+u[i,w]
	return
	
main()
