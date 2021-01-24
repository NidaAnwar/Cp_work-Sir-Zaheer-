# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:13:59 2019

@author: hp.840.G1
"""

#def f(x):
#   return (x**3-x-1)
#epsilon=0.001
#a=0
#while a==0:
#    x1=float(input("enter x1 =  "))
#    x2=float(input("enter x2 =  "))
#    if(f(x1)*f(x2)<0):
#        a=1
#    else:
#        print("root does not exist between given value of x1 and x2")
#        
#xm=x1
#print("x1  ,x2  , f(x1)   , f(x2), x3   f(x3)")
#while (abs(f(x1))>epsilon):
#    xm=(x1+x2)/2
#    if (f(xm)==0):
#        print(" root is = ",x1)
#        break
#    print(round(x1,3),round(x2,3),round(f(x1),3), round(f(x2),3),round(xm,3), round(f(xm),3))
#    if(f(x1)*f(xm)<0):
#        x2=xm
#    else:
#        x1=xm
#print("root is = ",xm)

#--------------- False position ---------------
#import math
#def f(x):
#   return (math.e*x**3-x-1)
#epsilon=0.001
#a=0
#while a==0:
#    x1=float(input("enter x1 =  "))
#    x2=float(input("enter x2 =  "))
#    if(f(x1)*f(x2)<0):
#        a=1
#    else:
#        print("root does not exist between given value of x1 and x2")
#
#xm=x1
#print("x1  ,x2  , f(x1)   , f(x2), x3   f(x3)")
#
#while (abs(f(xm))>epsilon):
#    xm=x1-((f(x1)*(x2-x1))/(f(x2)-f(x1)))
#    if (f(x1)==0):
#        print(" root is = ",x1)
#        break
#    print(round(x1,3),round(x2,3),round(f(x1),3), round(f(x2),3),round(xm,3), round(f(xm),3))
#    if(f(x1)*f(xm)<0):
#        x2=xm
#    else:
#        x1=xm
#print("root is = ",xm)

#---------------Secent Method

#def f(x):
#    return x**3-20
#E=0.001
#a=0
#while (a==0):
#    xo=float(input("enter xo ="))
#    x1=float(input("enter x1 = "))
#    a=1
#print("xo, x1 , x2 ,f(xo) ,f(x1) ,f(x2)") 
#
#x2=xo 
#while (abs(f(xo)>E)):
#    x2 = xo-((f(xo)*(xo-x1))/(f(xo)-f(x1)))
#    if  (abs(f(x2)<E)):
#        print(round(xo,5), round(f(xo),5), round(x1,5),round(f(x1),5),round(x2,5),round(f(x2),5))
#        xo=x2
#    else:
#        x1=xo
#        x2=x1
#print("root is x2 =",x2) 

#-----------Modified secent method------------- 

#import math
#xi=float(input('enter no xi =  '))
#e=0.001
#dx=0.001
#
#def f(x):
#    return x**2*math.sin(x)
#xo=xi
#print("x1    f(x1)    D     f(D)    xo     f(xo)")
#
#while (abs(f(xo))>e):
#    D=xi+(dx)
#    xo=xi-((dx*f(xi))/(f(D)-f(xi)))
#    print (round(xi,5), round(f(xi),5), round(D,5), round(f(D),5) ,round(xo,5),round(f(xo),5))
#    xi=xo
#print('The root is',xo)
#
#---------Newton Raphson method-----------------
#import math
#import sympy
#from sympy import Symbol
#from sympy import diff
#
#r=Symbol('r')
#E=0.001
#x1=float(input("enter no x1 = "))
#
#def f(x):
#    return math.e**x-x**3+1
#if (abs(f(x1))<E):
#    print("Root is =  ",x1)
#def fd(x):
#    z=diff(math.e**r-r**3+1) 
#    return z.subs(r,x)
#print("x1,   f(x1),   x2,    f(x2)")
#
#x2=x1
#while (abs(f(x2))>E):
#    x2=x1-f(x1)/(fd(x1))
#    print(round(x1,5),round(f(x1),5),round(x2,5),round(f(x2),5))
#    if (abs(f(x2))<E):
#        print("Root is =  ",x2)
#    else:
#        x1=x2 

#--------------- Matrix and Cramer's rule -----------

#import numpy as np 
#row =[]
#n=int(input('enter no of unknown :'))
#[row.append(int(input('enter xi :'))) for i in range (0,n) for j in range (0,n+1)]
#matrix=np.array(row).reshape(n,n+1)
#print(matrix)

#det_D=np.linalg.det(matrix[:,:-1])
#if det_D!=0:
#    for i in range(0,n):
#        matrix=np.array(row).reshape(n,n+1)
#        matrix[:,1:i+1]=matrix[:,-1:]
#        xi = (np.linalg.det(matrix[:,:-1]))/det_D
#        print("x",i,"=",xi)
#else :
#    print('solution is not possible')

#-------------Newton forward ----------------

#n = int(input("enter total number of elements"))
#x,y=[],[]
#for i in range(0,n):
#    x.append(float(input("enter elements of x: ")))
#    y.append(float(input("enter elements of y :")))
#b = []
#b.append(y)
#for i in range(0,n-1):
#    a = []
#    for j in range(0,n-1):
#         a.append(y[j+1]-y[j])
#    b.append(a)
#    n-=1
#    y=a
#print(a)
#print("==========formula-part============")
#m = len(x)
#value = float(input("y(x) : enter x: "))
#p = (value- x[0])/(x[1] - x[0])
#Temp = []
#import math
#temp = 1
#for i in range(0,m-1):
#    temp = temp *(p-i)
#    Temp.append(temp)
#umair = b[0][0]
#for i in range(0,m-1):
#    umair = umair + ((Temp[i]*b[i+1][0])/(math.factorial(i+1)))
#print(f"\nvalue at  f({value}) is , {round(umair,5)}")

# --Newton Backward
#n = int(input("enter number of elements: "))
#b =[]
#b.append(y)
#for i in range(0,n-1):
#    a = []
#    for j in range(1,n):
#        a.append(y[j]-y[j-1])
#    b.append(a)
#    n-=1
#    y=a
#print(b)
#print("+=+=+=+=+=+formula-part=+=+=+=+=+")
#m = len(x)
#value = float(input("y(x) : enter x: "))
#p = (value - x[-1])/(x[1]-x[0])
#temp = 1
#Temp = []
#import math
#for k in range(0,m-1):
#    temp = temp * (p+k)
#    Temp.append(temp)
#umair = b[0][-1]
#for l in range(0,m-1):
#    umair = umair + ((Temp[l]*b[l+1][-1])/(math.factorial(l+1)))
#print(f"the value of f({value}) is {umair}")

#-----------Lagrange Interpolation---------

#x,y= [1,3,5,7,9],[1,27,125,343,729]
#n = len(x)
#value = float(input("value of x "))
#a,b = [],[]
#for i in range(0,n):
#    b.append(value - x[i])
#a.append(b)
#for i in range(0,n):
#    xi = []
#    for j in range(0,n):
#        xi.append(x[i]-x[j])
#    a.append(xi)    
#print(a)
#print("=========formula-part=========")
#f_x = []  
#for i in range(0,n):
#    temp = 1
#    for j in range(0,n):
#        if i != j:
#            temp = temp * (a[0][j]/a[i+1][j])
#    f_x.append(temp*y[i])
#f_x
#sum = 0
#for i in range(0,n):
#    sum = sum + f_x[i]
#print(f'f({value}) = {sum}')

#----------GENERAL PROGRAM
#x = []
#f_x = []
#n = int(input("enter total number of elements: "))
#for i in range(0,n):
#    x.append(int(input(" enter elements of x: ")))
#    f_x.append(int(input(" enter elements of f_x: ")))
#b = []
#b.append(f_x)
#for i in range(0,n-1):
#    a = []
#    for j in range(1,n):
#        a.append((f_x[j] - f_x[j-1])/(x[j+i]-x[j-1]))
#    b.append(a)
#    n -=1
#    f_x = a    
#print(b
#    )
#print("::::::::::: formula part :::::::::::")
#m = len(x)
#Temp = []
#value = int(input("f(x) : enter value of x: "))
#Temp = []
#temp = 1
#for i in range(0,m-1):
#    temp = temp * (value - x[i])
#    Temp.append(temp)
#sum = 0
#for i in range(0,m-1):
#    sum = sum +  ((Temp[i]*b[i+1][0]))
#sum = sum + b[0][0]
#print(f"f({value}) = {sum}" )

        
#  monticarlo 
#import matplotlib.pyplot as plt
#n = 1000000
#z,y = [],[]
#for i in range(0,n):
#    x1 = np.random.rand()
#    x2 = np.random.rand()
#    if (x2**2 + x1**2 <1):
#        z.append(x1)
#        y.append(x2)
#estimate = (4*len(z))/n
#print(estimate)
#plt.figure(figsize = (4,4))
#plt.scatter(z,y,c="green")
#plt.show()


#---Monticarlo integration-----------
#import numpy as np
#a = 0
#b = 2
#def f(x):
#    return(x**2 - 2*x + 15)
#x1 = np.random.rand()
#x2 = np.random.rand()
#x3 = np.random.rand()
#x4 = np.random.rand()
#x5 = np.random.rand()
#area = 1/n*(f(x1)+f(x2)+f(x3)+f(x4)+f(x5))*(b-a)
#print(area)
#def f1(x):
#    return (x**3)/(3-x**2 + 15*x)
#print(f1(b)-f1(a))


#--------Gauss Elimination Method-----------------
#
#Solve the following equations using 
#Gauss Elimination Method 
#4x+3y+1=0 and -y+2x=3
#
#import numpy as np
#n=int(input("enter # of unknown = "))
#col=[]
#for i in range (0,n):
#    row=[]
#    for j in range (0,n+1):
#        print("enter elements of A =",i,j)
#        x=int(input())
#        row.append(x)
#    col.append(row)
#
#A=np.matrix(col)        
#print("A=",A)         
#
#for i in range(0,n):
#    for j in range (i+1,n):
#        factor =A[j,i]/A[i,i]
#        for k in range (0,n+1):
#                A[j,k]=A[j,k]-factor*A[i,k]
#
#print("A'=",A)  
#a=np.zeros(n)
#a[n-1]=(A[n-1,n]/A[n-1,n-1])
#for i in range (n-2,-1):
#    sum=0
#    for j in range(i+1,n):
#        sum=sum+A[i,j]*a[j]
#        a[i]=(A[i,n]-sum)/A[i,i]
#print ("a=",a)         
#

#------------Gauss Jordon--------------------

##Solve the following equations using Gauss Jordon Method 
##4x+3y-1=0 and 2x+3y-4=0
#    
##import numpy as np
##n=int(input("enter # of unknown = "))
##col=[]
##for i in range (0,n):
##    row=[]
##    for j in range (0,n+1):
##        print("enter elements of A =",i,j)
##        x=int(input("coef ="))
##        row.append(x)
##    col.append(row)
##
##A=np.matrix(col)        
##print('A:')
##print(A)
##      
##for i in range(0,n):
##    for j in range(0,n):
##        A[i,:]=A[i,:]/A[i,i]
##        if (i!=j):
##            factor=A[j,i]/A[i,i]
##            for k in range (0,n+1):
##                A[j,k]=A[j,k]-factor*A[i,k]
##print("A':")
##print(A)
##
##print("x,y",A[:,n]) 
#
## ===========Least square fitting-==============
#
## Ques :Find the least square regression line y = a x + b
##x	0	1	2	3	4
##y	2	3	5	4	6
#
##import matplotlib.pyplot as plt
##
##x= []
##y= []
##[sumx,sumy,sumxy,sumxx] = [0,0,0,0]
##n = int(input("enter total number of elements: "))
##for i in range(0,n):
##    x.append(int(input("enter elements of x:")))
##    y.append(int(input("enter elements of y:")))
##    sumx+=x[i]
##    sumy+=y[i]
##    sumxy+=x[i]*y[i]
##    sumxx+=x[i]*x[i]
##    
##b = (n*sumxy - sumx*sumy)/ (n*sumxx - sumx**2)
##a=((sumy - b*sumx)/n)
##print('a:',a)
##print('b:',b)
##
##plt.plot(x,y)
##-------------------------------------------------------
##Ques #2) Find the least square regression line  : y = ax**b
##x	1	3	5	7	9
##y	2	4	6	8	10
#
##import matplotlib.pyplot as plt
##import numpy as np
##from numpy import log as ln
##x= []
##y= []
##[sumx,sumy,sumxy,sumxx] = [0,0,0,0]
##n = int(input("enter total number of elements: "))
##for i in range(0,n):
##    x.append(int(input("enter elements of x:")))
##    y.append(int(input("enter elements of y:")))
##    X=ln(x)
##    Y=ln(y)
##    sumx+=X[i]
##    sumy+=Y[i]
##    sumxy+=X[i]*Y[i]
##    sumxx+=X[i]*X[i]
##    
##B = (n*sumxy - sumx*sumy)/ (n*sumxx - sumx**2)
##A=((sumy - B*sumx)/n)
##a = np.e**A
##
##print("A:",A)
##print("b:",B)
##
##print("-------------------")
##print("a =", a)
##
##ynew=[]
##for j in range(0,n):
##    w = a*x[j] **B
##    ynew.append(w)
##
##plt.plot(x,ynew)
##plt.plot(x,y)
##---------------------------------------------------
##--Q no 3:Find the Least square fitting line if  y = ab**x
##x	0	1	2	3	4
##y	2	3	5	4	6
#
##import matplotlib.pyplot as plt
##import numpy as np
##from numpy import log as ln
##x= []
##y= []
##[sumx,sumy,sumxy,sumxx] = [0,0,0,0]
##n = int(input("enter total number of elements: "))
##for i in range(0,n):
##    x.append(int(input("enter elements of x:")))
##    y.append(int(input("enter elements of y:")))
##    Y=ln(y)
##    sumx+=x[i]
##    sumy+=Y[i]
##    sumxy+=x[i]*Y[i]
##    sumxx+=x[i]*x[i]
##    
##B = (n*sumxy - sumx*sumy)/ (n*sumxx - sumx**2)
##A=((sumy - B*sumx)/n)
##print ("----------------")
##print("A=",A,"B=",B)
##print("-------------------")
##
##a = np.e**A
##b = np.e**B
##print("a =", a)
##print("b =", b)
##ynew=[]
##for j in range(0,n):
##    w = a*b **x[i]
##    ynew.append(w)
##
##plt.plot(x,ynew)
##plt.plot(x,y) 
##----------------- Quardatic fitting --------------------
#import numpy as np
#import matplotlib.pyplot as plt
#x,y,ynew= [],[],[]
#[sumx,sumy,sumxy,sumx2,sumx3,sumx4,sumyx2] = [0,0,0,0,0,0,0]
#n = int(input("enter total number of elements: "))
#for i in range(0,n):
#    x.append(float(input("enter elements of x:")))
#    y.append(float(input("enter elements of y:")))
#    sumx+=x[i]
#    sumy+=y[i]
#    sumxy+=x[i]*y[i]
#    sumx2+=x[i]*x[i]
#    sumx3+=x[i]*x[i]*x[i]
#    sumx4+=x[i]*x[i]*x[i]*x[i]
#    sumyx2+=x[i]*x[i]*y[i]
#
#A=np.array([[n,sumx,sumx2],[sumx,sumx2,sumx3],[sumx2,sumx3,sumx4]])
#print('A',A)
#
#Ain=(np.linalg.inv(A))
#B=np.array([[sumy],[sumxy],[sumyx2]])
#print('B:',B)
#a,b,c =(np.round(np.matmul(Ain,B),1))
#print("a,b,c=",a,b,c)
#for j in range(0,n):
#    w = a*x[j]**2+b*x[j]+c
#    ynew.append(w)
#plt.plot(x,ynew,"red")
#plt.plot(x,y)

##============Trapozoid=================
#
##xo=int(input("enter no:"))
##x1=int(input("enter no:"))
##def fn(x):
##    return x**3-x**2+5
##I=(x1-xo)*(fn(xo)+fn(x1))/2
##print(I)
#
##----MULTIPLE TRAPOZOID -----------------
#import math
#xo=1
#xn=3.5
#n=10
#h=(xn-xo)/n
#def fn(x):
#    return math.e**x-x**2+5
#sum=0
#for i in range (1,n):
#    x=xo+i*h
#    sum=sum+2*fn(x)
#I=(fn(xo)+sum+fn(xn))*h/2*n
#print(I)
#

##-------------Simpson Method-----------
##xo=int(input("enter xo:"))
##x1=int(input("enter xn:"))
##xm=(x1+xo)/2
##n=int(input("enter n:"))
##h=(x1-xo)/n
##def fn(x):
##    return x**3-x**2+5
##I=(x1-xo)*(fn(xo)+4*fn(xm)+fn(x1))/6
##print('I:',I)

##----------Numerical D.E-----------------
#import matplotlib.pyplot as plt
#import math
#x=int(input("enter x:"))
#y=int(input("enter y:"))
#h=0.25
#A=[]
#B=[]
#C=[]
#for i in range(0,20):
#    def f(x,y):
#        return math.e**x/y
#    xi=x+h
#    yi=y+(h*f(x,y))
#    print(xi)
#    print(yi)
#    x=xi
#    y=yi
#    A.append(xi)
#    B.append(yi)
#    C.append((math.e**x-0.21)**0.5)
#print(A)
#print(B)
#plt.plot(A,B,'-')
#print("-- Orignal VS Numerical--")
#plt.plot(A,C,'.')

















