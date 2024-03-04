import numpy as np 
from numpy.linalg import inv


def f1(x,y):
    return x**2+x*y-10
def f2(x,y):
    return y+3*x*(y**2)-57


def derive_x(function, value1,value2):
    h = 1e-06
    top = function(value1 + h,value2) - function(value1,value2)
    bottom = h
    slope=top / bottom
    # Returns the slope to the third decimal
    return slope.item()

def derive_y(function, value1,value2):
    h = 1e-06
    top = function(value1,value2 + h) - function(value1,value2)
    bottom = h
    slope = top / bottom
    # Returns the slope to the third decimal
    return slope.item()



def main():
    #Initial guess values of y1 and y2 at time = 0. This is constant for each time step
    y1_o = 1.5
    y2_o = 3.5
    #Column vector to store the values for vectorization of the code. This changes during each iteration at that time step
    Y_old = np.zeros((2,1))
    Y_old[0] = y1_o
    Y_old[1] = y2_o
    F = np.copy(Y_old)
    Y_new = np.zeros((2,1))
    J = np.zeros((2,2))

    file = open(f'./Output2d/{y1_o}-{y2_o}.txt', "w")
    cheke_loop=0
    while (cheke_loop==0):
        F[0] = f1(Y_old[0],Y_old[1])
        F[1] = f2(Y_old[0],Y_old[1])
        #Row 1
        J[0,0] = derive_x(f1,Y_old[0],Y_old[1])
        J[0,1] = derive_y(f1,Y_old[0],Y_old[1])
        #Row 2
        J[1,0] = derive_x(f2,Y_old[0],Y_old[1])
        J[1,1] = derive_y(f2,Y_old[0],Y_old[1])
        Y_new = Y_old - np.matmul(inv(J),F)
        if Y_old[0].item()==Y_new[0].item() and Y_old[1].item()==Y_new[1].item():
            cheke_loop=1
        print(Y_old)
        file.write(f'{Y_old[0].item()}\t{Y_old[1].item()}\n')
        Y_old = Y_new
    file.close()
    pass


if __name__=="__main__":
    main()