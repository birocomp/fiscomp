import numpy as np

def init():
	nx,ny=(401,401) #ODD NUMBERS to ensure (x,y)=(0,0) is in the grid
	dt, dx=(0.25,2) #time-step (a.u.); spatial-step (a.u.)
	nmax, nf= (4000,300) #max time iterations; number of frames to save
	v=1.0 #wave speed
	e=np.e;
	dx2=1.0/(dx**2); #for optimization (multiplication has less comp. cost than div)
	nt=0;

	x=np.linspace(-(nx+1)*dx/2, (nx+1)*dx/2, nx+2) #array with x coords
	y=np.linspace(-(ny+1)*dx/2, (ny+1)*dx/2, ny+2) #array with y coords
	X, Y=np.meshgrid(x, y)	#defines the grid

	if nmax%nf==0:
		fr=np.int(nmax/nf)
	else:
		fr=nmax//(nf)

	t=np.linspace(0, nmax*dt-dt, nmax)	#array with time coords
	tfr=t[0::fr]#picks time values from 't' every 'fr'

	V=np.zeros((nx+2,ny+2), dtype=np.float64)
	u=np.zeros((nx+2,ny+2), dtype=np.float64)
	ut=np.zeros((nx+2,ny+2), dtype=np.float64)
	ut1=np.zeros((nx+2,ny+2), dtype=np.float64)
	U=np.zeros((nx+2,ny+2,nf), dtype=np.float64)
	
	return nx,ny,dt,dx,nmax,nf,v,e,dx2,nt,x,y,X,Y,fr,t,tfr,V,u,ut,ut1,U
