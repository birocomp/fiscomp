import matplotlib.pyplot as plt
import numpy as np

a=np.array([46.15203, 0.640044, 0.236, 0.052, 0.12180, 0.610216])
texto=['Python','Matlab','gfortran','ifort','Java', 'Python\n+ \n Numpy']
b=np.array([1,2,3,4,5,6])
fig, ax = plt.subplots()
plt.ylabel('Tempo (s)')
ax.set_ylim([0,2])
plt.bar(b,a,tick_label=texto)
plt.show()