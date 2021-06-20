import gvars
import numpy as np
from numba import cuda,jit

nx,ny,dt,dx,nmax,nf,v,e,dx2,nt,x,y,X,Y,fr,t,tfr,V,u,ut,ut1,U=gvars.init()

def oneslit(C,WW,SH,horizontal=False):
	print('Defining Potential...')
	cx, cy=C
	WH=y.max()-y.min()-100.0
	if horizontal==False:
		for i in range(1,nx+1):
			for j in range(1,ny+1):
				if np.abs(y[j]-cx)<=WW/2:
					V[i,j]=1.0
					if np.abs(x[i])<=WW/2:
						V[i,j]=0
	else:
		for i in range(1,ny+1):
			for j in range(1,nx+1):
				if np.abs(x[i])<=WW/2:
					V[i,j]=1.0
					if np.abs(y[j])<=WW/2:
						V[i,j]=0
	return np.abs(V-1.0)
	print('Potential Defined')
	
def twoslit(C,WW,SH,SS,horizontal):
	print('Defining Potential...')
	cx, cy=C
	WH=y.max()-y.min()-100.0
	if horizontal==False:
		for i in range(1,nx+1):
			for j in range(1,ny+1):
				if np.abs(y[j]-cx)<=WW/2:
					V[i,j]=1.0
					if np.abs(x[i]-SS)<=WW/2:
						V[i,j]=0
					if np.abs(x[i]+SS)<=WW/2:
						V[i,j]=0
	else:
		for i in range(1,ny+1):
			for j in range(1,nx+1):
				if np.abs(x[i]-cx)<=WW/2:
					V[i,j]=1.0
					if np.abs(y[j]-SS)<=WW/2:
						V[i,j]=0
					if np.abs(y[j]+SS)<=WW/2:
						V[i,j]=0
						
	return np.abs(V-1.0)
	print('Potential Defined')
	
def square(S,C):
	cx, cy=C
	for i in range(1,nx+1):
		for j in range(1,ny+1):
			if np.abs(x[i]-cx)<S/2:
				if np.abs(y[j]-cy)<S/2:
					V[i,j]=1.0
	return np.abs(V-1.0)
	
def circle(V,R,C,IR):
	cx, cy=C
	U = V*0
	U = (X-cx)**2+(Y-cy)**2
	U = np.logical_and(U > (IR**2), U < (R**2))*1.0
	V = np.abs(V-1.0)+np.abs(U-1.0)
	return V
			
# U V Vf
# 1 1 1
# 0 1 0
# 0 0 0
# 1 0
