# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 02:57:08 2019

@author: M Qasim_2
"""

#-----Euler's Method------------
#import matplotlib.pyplot as plt
#x=1
#y=1.5
#n=20
#h=0.5
#A=[]
#B=[]
#C=[]
#
#print("x,  y , f(x,y) , xi, yi")
#def f(x,y):
#        return x/y
#def f1(x):
#    return ((x**2+ 5/4)**0.5)    
#    
#for i in range(0,n):
#    yi=y+(h*f(x,y))
#    x=x+h
#    y=yi
#    A.append(x)
#    B.append(yi)
#    C.append(f1(x))
#    print(x , y , f(x,y) , x+h , yi )    
#
#print(A)
#print(B)
#print(C)
#plt.plot(A,B,".")
#plt.plot(A,C,"--")

#--- RK Method (2nd order)
#import matplotlib.pyplot as plt
#x=1
#y=1.5
#n=20
#h=0.5
#A=[]
#B=[]
#C=[]
#D=[]
#k1=x/y
#
#print("x,  y , k1 , xi, yi , k2  ,yi+1")
#        
#def f1(x):
#    return ((x**2+ 5/4)**0.5)    
#    
#for i in range(0,n):
#    yi=y+(h*k1)
#    x=x+h
#    k2=(yi+1-yi)/h
#    yt=yi+(0.5*k1+0.5*k2)
#    y=yi
#    A.append(x)
#    B.append(yi)
#    D.append(yt)
#    C.append(f1(x))
#    
#    print(x , y , k1 , x+h , yi ,k2 ,yt )    
#
#print(A)
#print(B)
#print(C)
#plt.plot(A,B,".")
#plt.plot(A,C,"--")
#plt.plot(A,D,"*")


#----------Secent Method-------------
#def f(x):
#    return (8*x**3)-(6*x**2)-(261*x)+378
#
#E=0.00001
#a=0
#while (a==0):
#    xo=float(input("enter xo ="))
#    if (abs(f(xo))<=E):
#        print("root is xo =",xo)
#        break
#    x1=float(input("enter x1 = "))
#    if(abs(f(x1))<=E):
#        break
#        print("root is x1 =",x1)
#    else:
#        a=1
#x2=xo
#print("xo, x1 , x2 ,f(xo) ,f(x1) ,f(x2)") 
#print("---------------------------------") 
#    
#while (abs(f(x2)>E)):
#        x2 = xo-((f(xo)*(xo-x1))/(f(xo)-f(x1)))
#        if((f(x2))<=E):
#             print (round(xo,5), round(f(xo),5), round(x1,5),round(f(x1),5),round(x2,5),round(f(x2),5))
#             xo=x2
#        else:
#           x1=xo
#           x2=x1
#print("-------------")
#print("root is x2 =",x2)   
