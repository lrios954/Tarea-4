import sys
import os
import pylab
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.optimize import curve_fit


angulo=[]
Id=[]
gravedad=[]
gravedad_prom=[]

variaciones=[]

tablafinal=[]


datosFinales=open("data.dat", "w")

path="ComputationalPhysicsUniandes/homework/hw4_data/"
dirs=os.listdir(path)

for i in range(1):
	Data=np.loadtxt("ComputationalPhysicsUniandes/homework/hw4_data/"+dirs[i])


for Data in os.listdir(path):

#Nombre con el ID, angulo, etc (arraylist)
	nombre= (Data.replace(".dat","")).split("_")
#Variables en el documento
	
	Id.append(nombre[1])
	angulo.append(nombre[3])

	tiempo=[]
	pos_y=[]
	pos_x=[]
	info=[]
	arch =open(path+Data)
	for line in arch:

		info.append(line.split())

	info.remove(['#','time','pos_x','pos_y'])
	info.remove(['#', '[second]', '[meter]', '[meter]'])
	for line in info:
		
		tiempo.append(float(line[0]))
		pos_x.append(float(line[1]))
		pos_y.append(float(line[2]))
	

#HASTA AQUI ESTAN TODOS LOS DATOS!!

	#ajuste de polinomios de grado 1 y 2
	x_fit = numpy.polyfit(numpy.array(tiempo), numpy.array(pos_x), 1.0)
	y_fit = numpy.polyfit(numpy.array(tiempo), numpy.array(pos_y), 2.0)

	

#######

#tablafinal.append((int(nombre[1]),float(nombre[3]),float(x_fit[0]),float(y_fit[1]),(-0.5*float(y_fit[0])))) Codigo en revision

	
	tablafinal.append((int(nombre[1]),float(nombre[3]),float(x_fit[0]),float(y_fit[1]),(-0.5*float(y_fit[0]))))
	


tablafinal.sort(key=lambda tup: tup[1])


Data.write('\n'.join('%f %f %f %f %f' % x for x in table))



for line in table:

	theta.append(line[1])
	pca.append(repr(line[2])+' '+repr(line[3])+' '+repr(line[4]))


n_dimensions=3
n_measurements=len(pca)

data=numpy.empty((n_dimensions,n_measurements,)) ###Este data no se repite?####################################

for i in range(n_measurements):

    datosLinea=pca[i].split()

    for j in range(n_dimensions):
        
        datosLinea[j]=float(datosLinea[j])
        data.itemset((j,i),datosLinea[j])

covariance_matrix=numpy.cov(data)

w,u = eigh(covariance_matrix, overwrite_a = True)

#Imprime valores y vectores propios 

out_name='pca.dat'
out=open(out_name,'w')
out.write('Eigenvalues:')
out.write('\n')
out.write(''.join('%s' % w[::-1]))
out.write('\n')
out.write('Eigenvectors:')
out.write('\n')
out.write(''.join('%s' % u[:,::-1]))
out.close()

print 'PCA results exported sucessfully in pca.dat'

angles = list(set(theta))

# calcula la gravedad promedio 
for ang in angulo:
	
	promedio = 0.0
	
	for line in tablafinal:
	
		if (line[1] == ang):
		
			promedio += line[4]
		#####################################Creo que hay que revisar esta indentacion###########	
		promedio /= 6.0
		gravedad_prom.append(promedio)
	
# graficas 
pylab.plot(angles, avg_grav, '.')
pylab.xlabel('angulo(grados)')
pylab.ylabel('Gravedad promedio(m/s2)')
pylab.title('Gravedad/Angulo')
			#pylab.savefig('gravityplot')
			#pylab.grid(True)

pylab.show()



for i in range(len(gravedad_prom)):

	variaciones.append(1-(gravedad_prom[i]/(9.81)))

# Ajuste a funcion seno + constante
def f(x, a, b):
     return a*numpy.sin(numpy.deg2rad(x)) + b

parameters = (scipy.optimize.curve_fit(f, numpy.array(angulo), numpy.array(variaciones)))[0]

pylab.plot(angulo, f(angulo,parameters[0],parameters[1]), '.')
pylab.plot(angulo, variaciones, '.')
pylab.xlabel('Angulos(grados)')
pylab.ylabel('Variaciones en gravedad')
pylab.title('Variaciones en g/angulo')
		#pylab.grid(True)
pylab.show()


pylab.plot(angulos, f(angulos,parameters[0],parameters[1])-variaciones, '.')
pylab.xlabel('Angulo(grado)')
pylab.ylabel('Residuo')
pylab.title('Residuo/Angulo')
		#pylab.savefig('residueplot')
		#pylab.grid(True)
pylab.show()









