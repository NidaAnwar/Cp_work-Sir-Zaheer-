## -*- coding: utf-8 -*-
#"""
#Created on Fri Oct  4 22:42:44 2019
#
#@author: M Qasim_2
#"""
#
##----Bisection Method-------------------
#
## Ques : The eq f(x) = 2-x2*sin(x) =0  has a sol in the intervel (-1,2),using bisection method
##find out the root and no of iterations needed for accuracy 10**-6 
##
##import math
##def f(x):
##   return (x**3 -7*x+1)
##a=0
##while a==0:
##    x1=float(input("enter x1 =  "))
##    if (f(x1)==0):
##        print(" root is x1 = ",x1)
##        break
##    x2=float(input("enter x2 =  "))
##    if (f(x2)==0):
##        print(" root is x2 = ",x2)
##        break
##    if(f(x1)*f(x2)<0):
##        a=1
##    else:
##        print("root does not exist between given value of x1 and x2")
##xm=x1
##epsilon=0.001
##print("x1  ,x2  , f(x1)   , f(x2), x3,  f(x3)")
##print("==========================================")
##while (abs(f(xm))>epsilon):
##    
##    xm= (x1+x2)/2
##    if (f(x1)==0):
##        print(" root is = ",x1)
##        break
##    print(round(x1,3),round(x2,3),round(f(x1),3), round(f(x2),3),round(xm,3), round(f(xm),3))
##    if(f(x1)*f(xm)<0):
##        x2=xm
##    else:
##        x1=xm
##print("==========================================")
##print("root is = ",xm)
#
##---------------False position method---------------------
#
## Ques : The eq f(x) = e**-x - cos(x) =0  has a sol in the intervel (2,7) ,using false method
##find out the root needed for accuracy 10**-7 
#
##import math
##
##def f(x):
##   return (math.e*x**3-x-1)
##a=0
##while a==0:
##    x1=float(input("enter x1 =  "))
##    if (f(x1)==0):
##        print(" root is x1 = ",x1)
##        break
##    x2=float(input("enter x2 =  "))
##    if (f(x2)==0):
##        print(" root is x2 = ",x2)
##        break
##    if(f(x1)*f(x2)<0):
##        a=1
##    else:
##        print("root does not exist between given value of x1 and x2")
##xm=x1
##epsilon=0.001
##print("x1  ,x2  , f(x1)   , f(x2), x3   f(x3)")
##print("==========================================")
##while (abs(f(xm))>epsilon):
##    
##    xm=x1-((f(x1)*(x2-x1))/(f(x2)-f(x1)))
##    if (f(x1)==0):
##        print(" root is = ",x1)
##        break
##    print(round(x1,3),round(x2,3),round(f(x1),3), round(f(x2),3),round(xm,3), round(f(xm),3))
##    if(f(x1)*f(xm)<0):
##        x2=xm
##    else:
##        x1=xm
##print("==========================================")
##print("root is = ",xm)
#
#
#
##-------------Secent Method---------------
#
##Quest : find value of x if f(x)=x3-20=0  using secent method for 3 iteration , where xo=4 x1=5.5 
##
##def f(x):
##    return x**3-20
##
##E=0.001
##a=0
##while (a==0):
##    xo=float(input("enter xo ="))
##    if (abs(f(xo))==E):
##        print("root is xo =",xo)
##        break
##    x1=float(input("enter x1 = "))
##    if(abs(f(x1))==E):
##        print("root is x1 =",x1)
##        break
##    else:
##        a=1
##
##x2=xo 
##print("xo, x1 , x2 ,f(xo) ,f(x1) ,f(x2)") 
##print("---------------------------------")
##   
##while (abs(f(x2)>E)):
##    x2 = xo-((f(xo)*(xo-x1))/(f(xo)-f(x1)))
##    if (abs(f(x2)>E)):
##        print(round(xo,5), round(f(xo),5), round(x1,5),round(f(x1),5),round(x2,5),round(f(x2),5))
##        xo=x2
##    else:
##        x1=xo
##        x2=x1
##
##print("-------------")
##print("root is x2 =",x2)  
#
##-----------Modified secent method------------- 
##use the modified secent method to find the roots of the f(x)=xtan(x) ,xo=1 dx=.001
#
##import math
##xi=float(input('enter no xi =  '))
##e=0.001
##dx=0.001
##
##def f(x):
##    return x**2*math.sin(x)
##xo=xi
##print("x1    f(x1)    D     f(D)    xo     f(xo)")
##print("=====================================")
##while (abs(f(xo))>e):
##    D=xi+(dx)
##    xo=xi-((dx*f(xi))/(f(D)-f(xi)))
##    print (round(xi,5), round(f(xi),5), round(D,5), round(f(D),5) ,round(xo,5),round(f(xo),5))
##    xi=xo
##print("=====================================")
##print('The root is',xo)
#
##---------Newton Raphson method-----------------
##import math
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
#x2=x1
#print("x1,   f(x1),   x2,    f(x2)")
#print("============================")
#
#while (abs(f(x2))>E):
#    x2=x1-f(x1)/(fd(x1))
#    print(round(x1,5),round(f(x1),5),round(x2,5),round(f(x2),5))
#    if (abs(f(x2))<E):
#        print("============================")
#        print("Root is =  ",x2)
##
##    else:
##        x1=x2 
#
##------------------Matrix Operation-----------------------
##Dterine the deteminant ,transpose ,inverse and product of two matrices?
##A=[2,5,2],[1,5,0],[7,8,2]
##B=invers of (A)
#
##
##import numpy as np 
##n=int(input("enter # of unknown="))
##col=[]
##for i in range (0,n):
##    row=[]
##    for j in range(0,n):
##        print("enter elements of A ",i,j)
##        x=int(input("enter # ="))
##        row.append(x)
##    col.append(row)
##
##A=np.matrix(col)
##print("A =",A)
##
##B=np.linalg.inv(A)
##
##C=np.matmul(A,B)
##
##D=np.transpose(A)
##
##print('A:')
##print(A)
##
##print("  ")
##print('B:')
##print(B)
##
##print("   ")
##print('C:')
##print(C)
##
##print("   ")
##print('D:')
##print(D)
##
##print("    ")
##print('det A:')
##print(np.linalg.det(A))
##
##print("   ")
##print('det B:')
##print(np.linalg.det(B))
#
##------------------Cramers Rule-------------------------
#
## Ques: Solve the following eq and find the value of x and y by using cramer;s rule
## x-2y=5 ,3x-4y=8 
#    
##import numpy as np 
##n=int(input("enter # of unknown="))
##col=[]
##for i in range (0,n):
##    row=[]
##    for j in range(0,n):
##        print("enter elements of A ",i,j)
##        x=int(input("enter coef ="))
##        row.append(x)
##    col.append(row)
##A=np.matrix(col)
##
##B=[]
##for k in range(0,n):
##    x1=int(input("enter coef = "))
##    B.append(x1)
##B=np.array(B)
##
##print("A:")
##print(A)
##
##print("B:")
##print(B)
##
##for w in range(0,n):
##    z=np.array(A)
##    for j in range (0,n):
##        z[j,i]=B[j]
##
##for w in range(0,n):
##    S=np.array(A)
##    for j in range (0,):
##        S[j,i-1]=B[j]
##
##print("z:")
##print(z)
##
##print('x=',np.linalg.det(z)/np.linalg.det(A))        
##print("S:")
##
##print(S)
##print('y =',np.linalg.det(S)/np.linalg.det(A))
#            
##--------------------Gauss Elimination Method-------------------------------
#
##Solve the following equations using Gauss Elimination Method 
##4x+3y+1=0 and -y+2x=3
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
##-----------------Gauss Jordon--------------------
#
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
#
#
##----------Newton Forward -----------
##import numpy as np
##x=[1,3,5,7,9]
##y=[1,27,125,343,729]
##n=len(x)
##z=y
##b=[]
##b.append(z)
##for i in range(0,n-1):
##    a=[]
##    for j in range(0,n-1):
##        a.append(z[j+1]-z[j])
##    b.append(a)
##    n=n-1
##    z=a
##print(b)
#
##----------Newton Backward-- -----------
##x=[2,4,6,8,10]
##y=[4,16,36,64,100]
##n=len(x)
##z=y
##b=[]
##b.append(z)
##for i in range(0,n-1):
##    a=[]
##    for j in range(0,n-1):
##        a.append(z[j+1]-z[j])
##    b.append(a)
##    n=n-1
##    z=a
##print(b)
#
##-----------------Divided difference---------------
##Generalize
#
##x=[1,2,3,4,5,6]
##y=[1,4,9,16,25,36]
##n=len(x)
##z=y
##b=[]
##b.append(z)
##
##
##for i in range(0,n-1):
##    a=[]
##    for j in range(0,n-1):
##        a.append((z[j+1]-z[j])/(x[j+1]-x[j]))
##    b.append(a)
##    n=n-1
##    z=a
##print(b)
#
###----------Lagrange--------------------------
#
##x=[2,3,4,7,12]
##y=[4,9,16,49,144]
##A,B,C,D,E=[],[],[],[],[]
##a=5
##for i in range (1,len(x)):
##    a1=(a-x[i-1])
##    A.append(a1)
##print("A=",A)
##gfrrg
##for j in range (1,len(x)):
##    b=(x[1]-x[j])
##    B.append(b)
##print("B=",B)
##
##for j in range (1,len(x)):
##    c=(x[2]-x[j])
##    C.append(c)
##print("C=",C)
##
##for i in range (1,len(x)):
##    d=(x[3]-x[i])
##    D.append(d)
##print("D=",D)
##
##for j in range (1,len(x)):
##    e=(x[4]-x[j])
##    E.append(e)
##print("E=",E)
#
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
##import math
##xo=1
##xn=3.5
##n=10
##h=(xn-xo)/n
##def fn(x):
##    return math.e**x-x**2+5
##sum=0
##for i in range (1,n):
##    x=xo+i*h
##    sum=sum+2*fn(x)
##I=(fn(xo)+sum+fn(xn))*h/2*n
##print(I)
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
#
##----------Numerical D.E-----------------
#
##import matplotlib.pyplot as plt
##import math
##
##x=int(input("enter x:"))
##y=int(input("enter y:"))
##h=0.25
##A=[]
##B=[]
##C=[]
##for i in range(0,20):
##    def f(x,y):
##        return math.e**x/y
##    xi=x+h
##    yi=y+(h*f(x,y))
##    print(xi)
##    print(yi)
##    x=xi
##    y=yi
##    A.append(xi)
##    B.append(yi)
##    C.append((math.e**x-0.21)**0.5)
##print(A)
##print(B)
##plt.plot(A,B,'-')
##print("-- Orignal VS Numerical--")
##plt.plot(A,C,'.')

#----------------Monticarlo------------
#-Q-calculate of pie its ratio circumference to radius..

#import matplotlib.pyplot as plt
#import random 
#n=10000
#z=[]
#y=[]
#for i in range (0,n):
#    x1=random.random()
#    x2=random.random()
#    if (x2**2 + x1**2 < 1):
#        z.append(x1)
#        y.append(x2)
#estimate=4*(len(z)) / n
#print ("E=",estimate)
#plt.figure(figsize=(4,4))
#plt.scatter(z,y, c="blue")
#plt.show()
#
##--
#import random
#a=0
#b=2
#def f(x):
#    return (x**2 - 2*x +15)
#x1= random.random()
#x2= random.random()
#x3= random.random()
#x4= random.random()
#x5= random.random()
#n=5
#area = (1/n)*(f(x1)+ f(x2)+ f(x3) +f(x4)+ f(x5))*(b-a)
#print ("area=",area)
#def f1(x) :
#    return (1/3)*x**3 -x**2 +15*x
#print ("integrated function",f1(b)-f1(a))
#
#
#
#import random
#a=0
#b=2
#n=100
##def f(x):
##    return (x**2) -( 2*x) +(15)
##for i in range (0,n):
##    xo=random.random()
##    xi=xo
##sum=0
##sum=sum+f(xo)
##area = (1/n)*(sum)*(b-a)
##print ("area=",area)
##def f1(x) :
##    return (1/3)*x**3 -x**2 +15*x
##print ("integrated function",f1(b)-f1(a))
###### if the limits are changed????
#
#import matplotlib.pyplot as plt
#import random
#a=5
#b=11
#n=100
#z=[]
#y=[]
#def f(x):
#    return (x**2) -( 2*x) +(15)
#for i in range (0,n):
#    xo=random.randrange(5,11,1)
#    xi=xo
##    z.append (xo)
##    y.append (f(xo))
##sum=0
##sum=sum+f(xo)
##area = (1/n)*(sum)*(b-a)
##print ("area=",area)
##def f1(x) :
##    return (1/3)*x**3 -x**2 +15*x
##print ("integrated function",f1(b)-f1(a))
##plt.figure(figsize=(4,4))
##plt.scatter(z,y, c="blue")
##plt.show()
#
#
#import matplotlib.pyplot as plt
#import random
#a=5
#b=11
#n=500
#z=[]
#y=[]
#def f(x):
#    return (x**2) -( 2*x) +(15)
#for i in range (0,n):
#    xo=a+((b-a)*(random.random()))
#    xi=xo
#    z.append (xo)
#    y.append (f(xo))
#sum=0
#sum=sum+f(xo)
#area = (1/n)*(sum)*(b-a)
#print ("area=",area)
#def f1(x) :
#    return (1/3)*x**3 -x**2 +15*x
#print ("integrated function",f1(b)-f1(a))
#plt.figure(figsize=(5,5))
#plt.scatter(z,y, c="blue")
#plt.show()
#
#
#
#
#
##import matplotlib.pyplot as plt
##import random
##a=5
##b=11
##n=500
##z=[]
#y=[]
#def f(x):
#    return (x**2) -( 2*x) +(15)
#for i in range (0,n):
#    xo=a+((b-a)*(random.random()))
#    xi=xo
#    z.append (xo)
#    y.append (f(xo))
#sum=0
#sum=sum+f(xo)
#area = (1/n)*(sum)*(b-a)
#print ("area=",area)
#def f1(x) :
#    return (1/3)*x**3 -x**2 +15*x
#print ("integrated function",f1(b)-f1(a))
#plt.figure(figsize=(5,5))
#plt.scatter(z,y, c="blue")
#plt.show()

#import matplotlib.pyplot as plt
#import random 
#n=100
#z=[]
#y=[]
#for i in range (0,n):
#    x1=random.random()
#    x2=random.random()
#    if (x2**2 + x1**2 < 1):
#        z.append(x1)
#        y.append(x2)
#estimate=4*(len(z)) / n
#print ("E=",estimate)
#plt.figure(figsize=(4,4))
#plt.scatter(z,y, c="blue")

#%% Integral of complex function by using 
#         Residual theorem
#
#import matplotlib.pyplot as plt
#from sympy import Symbol
#import numpy as np
#import math
#complex=(0+1j)
#z=Symbol ('z')
#A=[]
#B=[]
#def f(x):
#   return (x-xo)*((1-2*x**4)/(x*(x-1)*(x-2)*(x-3)*(x-4)))
#for i in range (0,5):
#    xo=(input("enter poles = "))
#    print(f(z))
#    a=f(z).subs(z,xo)
#    print(a)
#    A.append(a)
#    B.append(xo)
#sum=np.sum(A)
#print('sum of residue =',sum)
#print('I =' ,2*(math.pi)*complex*(sum))
#
#plt.figure(figsize=(2,2))
#plt.scatter(A,B, c="blue")

#import time 
#import tkinter as tkr 
#tk = tkr.Tk()
#canvas=tkr.Canvas(tk,width=50,height=800)
#canvas.grid()
#ball=canvas.create_oval(30,30,40,40,fill="Red")
#x=30
#y=30
#while True :
#    canvas.move(ball,x,y)
#    pos=canvas.coords(ball)
#    if pos[3]>=800 or pos[1]<=0:
#        y=-y
#    if pos[2]>=100 or pos[0]<=0:   
#        x=-x
#    tk.update()
#    time.sleep(0.02)
#    pass
#tk.mainloop()
#tk.down()


# Tangent and cot

#import numpy as np
#import matplotlib.pyplot as plt
#
#x= np.arange(0,10,.1);
#amplitude= np.tan(x)
#plt.plot(x,amplitude)
#
#x= np.arange(0,10,.1)
#amplitude= 1/np.tan(x)
#plt.plot(x, amplitude)
#
#plt.title('Waves')
#plt.xlabel('X')
#plt.ylabel('Amplitude' )
#plt.grid(True)
#plt.axhline(y=0, color='k')

#import numpy as np
#import matplotlib.pyplot as plt
#x= np.arange(0,10,.2);
#amplitude= np.sin(x)
#plt.plot(x,amplitude)
#x= np.arange(0,20,.2);
#amplitude= np.cos(x)
#plt.plot(x, amplitude)
#plt.title('sine & cosine waves')
#plt.xlabel('X')
#plt.ylabel('Amplitude' )
#plt.grid(True)
#plt.axhline(y=0, color='k')


# Gauss quadrature 

#import math
# 
#def f(x):
#    return math.exp(-x**2)
#
#Ig1=2*f(0)
#Ig2=f(-0.577)+f(0.577)
#Ig3= (5/9)*(f(0.7745)+f(-0.7745))+(8/9)*f(0)
#    
#print(Ig1,Ig2,Ig3)



import matplotlib.pyplot as plot

import numpy as np

data = np.array([24.40,10.25,20.05,22.00,34.90,13.30])

plot.acorr(data, maxlags = 4)

plot.xlabel('Lag')

plot.ylabel('Autocorrelation')

 


























            




















