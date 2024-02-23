"""
Program to solve a system of non linear DE using the implicit Backward Euler method and multivariate NR method

dy1/dt = -0.04y1+10e4y2y3
dy2/dt = 0.04y1 -10e4y2y3 -3e7y2*y2
dy3/dt = 3e7y2*y2

Newton Raphson's multivariate method-
Y_new = Y_old -alpha*inv(J)*F(Y_old)

Implicit Euler's Method has been used (Backward)
i.e.
y1' = y1_n-y1_o = dt*RHS
=> f1 = y1_n-y1_o - dt*RHS
Similarly f2 and f3
"""

import numpy as np 
from numpy.linalg import inv
import matplotlib.pyplot as plt 

#Initial guess values of y1,y2 and y3 at time = 0. This is constant for each time step
y1_o = 1
y2_o = 0
y3_o = 0 

#Column vector to store the values for vectorization of the code. This changes during each iteration at that time step
Y_old = np.ones((3,1))
Y_old[0] = y1_o
Y_old[1] = y2_o
Y_old[2] = y3_o

#If we use = operator i.e. F = Y_old, when F changes in the loop, Y_old will also change, which we do not want.
#By using the copy function, Y_old doesn't change when F is updated in the loop

F = np.copy(Y_old)
Y_new = Y_old 

#Control values
tol = 1e-6
alpha = 1
t_start = 0
t_end = 10*60	#10 minutes 
dt = 1
n = int(t_end/dt)
t_steps = np.linspace(t_start,t_end,n+1)
#Storing the values at the end of each time step
y1_t = y1_o*np.ones((n+1,1))
y2_t = y2_o*np.ones((n+1,1))
y3_t = y3_o*np.ones((n+1,1))

def f1(y1_o,y1_n,y2_n,y3_n,dt):	#Defining the first non linear equation
	return y1_n-y1_o-dt*(-0.04*y1_n+10000*y2_n*y3_n)

def f2(y2_o,y1_n,y2_n,y3_n,dt):	#Defining the second non linear equation
	return y2_n-y2_o-dt*(0.04*y1_n-10000*y2_n*y3_n-3e07*pow(y2_n,2))

def f3(y3_o,y1_n,y2_n,y3_n,dt):	#Defining the third non linear equation
	return y3_n-y3_o-dt*(3e07*pow(y2_n,2))

def jacobian(y1_o,y2_o,y3_o,y1_n,y2_n,y3_n,dt):
	"""Returns the Jacobidan matrix using Forward differencing method
	Since the number of functions and variables are 3, thus the Jacobian matrix will be a 3x3 matrix
	"""
	h = 1e-06
	J = np.ones((3,3))
	#Row 1
	J[0,0] = (f1(y1_o,y1_n+h,y2_n,y3_n,dt)-f1(y1_o,y1_n,y2_n,y3_n,dt))/h
	J[0,1] = (f1(y1_o,y1_n,y2_n+h,y3_n,dt)-f1(y1_o,y1_n,y2_n,y3_n,dt))/h
	J[0,2] = (f1(y1_o,y1_n,y2_n,y3_n+h,dt)-f1(y1_o,y1_n,y2_n,y3_n,dt))/h

	#Row 2
	J[1,0] = (f2(y2_o,y1_n+h,y2_n,y3_n,dt)-f2(y2_o,y1_n,y2_n,y3_n,dt))/h
	J[1,1] = (f2(y2_o,y1_n,y2_n+h,y3_n,dt)-f2(y2_o,y1_n,y2_n,y3_n,dt))/h
	J[1,2] = (f2(y2_o,y1_n,y2_n,y3_n+h,dt)-f2(y2_o,y1_n,y2_n,y3_n,dt))/h

	#Row 3
	J[2,0] = (f3(y3_o,y1_n+h,y2_n,y3_n,dt)-f3(y3_o,y1_n,y2_n,y3_n,dt))/h
	J[2,1] = (f3(y3_o,y1_n,y2_n+h,y3_n,dt)-f3(y3_o,y1_n,y2_n,y3_n,dt))/h
	J[2,2] = (f3(y3_o,y1_n,y2_n,y3_n+h,dt)-f3(y3_o,y1_n,y2_n,y3_n,dt))/h

	return J

for t in range(1,n+1):
	itr = 1
	error = 1

	while (error>tol):
		#Calculating the Jacobian
		J = jacobian(y1_o,y2_o,y3_o,Y_old[0],Y_old[1],Y_old[2],dt)

		#Returning the values of the function at X_old
		F[0] = f1(y1_o,Y_old[0],Y_old[1],Y_old[2],dt)
		F[1] = f2(y2_o,Y_old[0],Y_old[1],Y_old[2],dt)
		F[2] = f3(y3_o,Y_old[0],Y_old[1],Y_old[2],dt)

		#Iteration update
		Y_new = Y_old - alpha*np.matmul(inv(J),F)

		error = max(abs(Y_new-Y_old))
		Y_old = Y_new
		print('time = {0}	iteration # = {1}	y1 = {2}	y2 = {3}	y3 = {4}'.format(t*dt,itr,Y_new[0],Y_new[1],Y_new[2]))
		itr = itr+1

	
	y1_t[t] = Y_new[0]
	y2_t[t] = Y_new[1]
	y3_t[t] = Y_new[2]
	#Updating the initial guess values for the next time step
	y1_o = Y_new[0]
	y2_o = Y_new[1]
	y3_o = Y_new[2]

#Plotting the results
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=15)
plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, pad=15)
plt.xticks(fontsize=31,color= '#262626')
plt.yticks(fontsize=31,color= '#262626')
plt.plot(t_steps,y1_t, linewidth = '10')
plt.plot(t_steps,y2_t, linewidth = '10')
plt.plot(t_steps,y3_t, linewidth = '10')
plt.xlabel('Number of time steps',fontsize=31,color= '#262626')
plt.ylabel('Variable values (y1,y2,y3)',fontsize=31,color= '#262626')
l = plt.legend(['y1','y2','y3'],fontsize=31)
for text in l.get_texts():
    text.set_color('#262626')
plt.title('Solution of system of ODEs using Multivariate Newton Raphson Method \n (Backward Euler Method)n Total time = {0} sec n Time step size = {1} sec Tolerance = {2}'.format(t_end,dt,tol),fontsize=31,color= '#262626')
plt.grid('on')
plt.subplots_adjust(top = 0.9, bottom=0.08, hspace=0.2, wspace=0.44)
plt.gcf().set_size_inches(22, 17)
plt.savefig("plot.png")