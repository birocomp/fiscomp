import numpy as np
import matplotlib.pyplot as plt

a=0.005
nt=100
t, dt=np.linspace(0,1000,nt,retstep=True)
u=np.zeros((nt))
u[0]=1.0

for i in range(0,nt-1):
    u[i+1]=u[i]+dt*a*u[i]
plt.plot(t,u)
plt.show()