# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 12:41:51 2019

@author: Nida Ali
"""

#-------- Python Basic , All lectures and Assignments--------

#-------------Variable ---------

#x="Muhammad"
#y="Qasim"
#z=x+y
#print (z)
#-------------------------------------------
#print x interms of len
#x="saba_anwar"
#print (len(x))  # len show the characters of variable or list
#---------------------------
#verifying type of character
#a=[1,2,3,5,7,2.4,3.8,4j] #a=list
#b=a[7]
#print(type(b))
#----------------------------------------------
#x=34e3
#print (x)
#------------------------------
#convert one type to another
#x = 1 # int
#y = 2.8 # float
#z = 1j # complex
#
##convert from float to complex:
#a = complex(y)
#
##convert from float to int:
#b = int(y)
#
##convert from int to complex:
#c = complex(x)
#
#print(a)
#print(b)
#print(c)
#
#print(type(a))
#print(type(b))
#print(type(c))
#Note :complex to int or comp to float can't possible
#--------------------------------------

#Q) Import the random module, and display a random number between 1 and 9
#import random
#print(random.randrange(1,10))
#---------------------------------

#Q)Get the character at position 1 (remember that the first character has the position 0):
#a = "Hello, World!"
#print(a[1])
#-------------------------------------

#Q) Get the characters from position 2 to position 5 (not included):
#b = "Hello, World!"
#print(b[2:5])
#-----------------------------

#Q) The lower() method returns the string in lower case:
#a = "Hello, World!"
#print(a.lower())
#-----------------------------------

# The lower() method returns the string in lower case:
#a = "Hello, World!"
#print(a.upper())
#--------------------------------------------

#Q) The replace() method replaces a string with another string:
#a = "Hello, World!"
#print(a.replace("H", "h"))
#------------------------------------

# using format for combine (str+int)
#x = 3
#y = 567
#z = 49.95
#myorder = "I want {} pieces of item {} for {} dollars."
#print(myorder.format(x, y, z))
#----------------------------------------------

#a=int(input ("enter a num a= "))
#b=int(input ("enter a num b= "))
#c=int(input ("enter a num c= "))
#d=a+b+c
#x="sum of {1}, {0} and {2} is equal to {3}"
#print (x.format(a,b,c,d))
##------------------------------

#a=67
#b=5
#x=a%b
#print (x)
#--------------------------

#a=67
#b=5
#x=a//b    #// round of ans
#print (x)

#---------Operator---------------------
#x=5 # assign vale
#x+=10    # x=x+10
#x-=10    # x=x-10
#x*=10    # x=x*10
#x/=10    # x=x/10
#x%=10    # x=x%10
#x&=10     # x=x&10
#print(x)
#----------------------------------

#a=int(input("enter a num ="))
#b=34
#c=45
#print(a<b or a>c)  #stat. is true or false
#----------------------------------

#a=int(input("enter a num a="))
#b=10
#c=float(input("enter a num c ="))
#print(a<c and a>b)
#--------------------------------------

#a=int(input("enter a num a="))
#b=10
#c=float(input("enter a num c ="))
#print (not(a<c and a>b))
#note if statement is true ans is false
# -----------------------------------------

#x = [2,4,6,8,10]
#y = [2,4,6,8,10]
#z = x
#print(x is z)
#
## returns True because z is the same object as x
#
#print(x is not y)
##
### returns False because x is not the same object as y, even if they have thew same content
##
#print(x == y)
#
#demonstrate the difference betweeen "is" and "==": this comparison returns True because x is equal to y
#------------------------------------

#Change the second item:
#thislist = ["apple", "banana", "cherry"]
#thislist[1] = "blackcurrant"
#print(thislist)
#--------------------------------

#Print all items in the list, one by one:
#a= ["apple", "banana", "cherry"]
#for i in a:
#  print(i)
#-------------------------------

#a = ["apple", "banana", "cherry"]
#for x in a:
#  print(x)
#---------------------------

#Using the append() method to append an item:
#x = ["apple", "banana", "cherry"]
#x.append("orange")
#print(x)
#-----------------------------------

#Insert an item as the second position:
#x = ["apple", "banana", "cherry"]
#x.insert(1,"orange")
#print(x)
#------------------------------

#The pop() method removes the specified index, (or the last item if index is not specified):
#thislist = ["apple", "banana", "cherry"]
#thislist.append("orange")
#print(thislist)
#------------------------------------
#The del keyword removes the specified index:

thislist = ["apple", "banana", "cherry"]
del thislist[0]
print(thislist)
#-------------------------------

#The clear() method empties the list:
#x= ["apple", "banana", "cherry"]
#x.clear()
#print(x)
#--------------------------

#Make a copy of a list with the copy() method:
#x = ["apple", "banana", "cherry"]
#y = x.copy()
#print(y)
#---------------------------------------------
#Make a copy of a list with the list method:
#x = ["apple", "banana", "cherry"]
#mylist = list(x)
#print(mylist)
#-------------------------------

#thislist = ["orange","apple", "banana", "cherry"]
#thislist.sort() #arranged in alphabatic odder
#print(thislist)
#----------------------------------------------------

#tuple =(1,2,3,4,5)
#print (tuple)
#-------------------------------------

#Loop through the set, and print the values:
#set = {"apple", "banana", "cherry"}
#
#for x in set:
#  print(x)
#---------------------------------------
#Add or update an item to a set, using the add() method:

#thisset = {"apple", "banana", "cherry"}
#
#thisset.update([3,5,8])
#
#print(thisset)
#Add multiple items to a set, using the update() method:

#x = {"apple", "banana", "cherry"}
#x.update(["orange", "mango", "grapes"])'
#print(x)
#-------------------------------------

##Create and print a dictionary:
##A ={
##    "Name":str(input("enter student name =")),
##    "Marks":int(input("enter marks =")),
##    "Standard" :str(input("enter class ="))
##    }
##print(A)

##-------------Mark Sheet--------------------------
#D={
#   "Name":str(input("enter name = ")),
#   "Class":str(input("enter class =")),
#   "uni":str(input("enter uni name =")),
#   "QM":int(input("enter QM no = ")),
#   "Nu":int(input("enter Nu no = ")),
#   "EPSo":int(input("enter EPSo no = ")),
#   "OPT" :int(input("enter OPT no = ")),
#   "Lab" :int(input("enter Lab no = ")),
#   }
#a=D["Name"]
#b=D["QM"]
#c=D["Nu"]
#d=D["EPSo"]
#e=D["OPT"]
#g=D["Lab"]
#Total=b+c+d+e+g 
#Grade = Total*100/500
#for i in D:
#    print (i,":",D[i])
#print ("Total Marks =" ,Total)
#print ("percentage  = ",Grade,"%" )
#if Grade >80 :
#    print (a, "got A+ ")
#elif Grade >70 :
#    print (a, "got A  ")
#elif (Grade > 60):
#        print(a," got B grade")
#elif (Grade <= 50):
#        print(a," is failed")    

#------------------------------------------------------------------

#---------------------Lecture # 01--------------------------------------------------

#print("nida")
#-------------------------------------------
#a=3
#b=6
#c=a+b
#print (c)
#---------------------------------------------------
#d="Ali"
#print (d)

# ------For statement -------------------------------------
#for i in range (1,11):
#    print (i,"Zaheer")
#-------------------------------
#for i in range (1,6):
#   print(5,'x',i,'=',5*i)
#----------------------------------
#for i in range (1,10):
#    sum=i
#    print(sum)
#    print("sum of first",i,"integer is =",sum)
#----------------------------------------------------------
#sum=0
#for i in range (1,10):
#     sum=i+sum
#     print(sum)
#     print("sum of first",i,"integer is =",sum)  
#---------------------------------------------------
#a=int(input("enter a num ="))
#print (a)
# ------------------------------------------

#q1=1.6*10**-19
#q2=1.6*10**-19
#k=9*10**9
#r=2
#F=k*q1*q2/r**2
#print(F)

#------ If and else  ------------------
#a=int(input("enter a no ="))
#if a%2!=0:
#    print ("a is odd no")
#else:
#    print ("a is even no")

#------------List------------------
#a=[1,2,3,4,5,6,7,8,9]
#b=a[6]
#print(a)
#print(b)
#-------------------------------------
#a=33
#b=10
#if(b>a):
#    print("b is greater than a")
#else:
#    print("b is not greater than equal to a ")

#--------Import -------------------------------
#import numpy as np
#a=np.array([1,2,3,5,7])
#b=a[3]
#print (a)
#print (b)

#---------- Elif -------------------------------
#marks = int(input("enter marks = "))
#if (marks > 80):
#    print ("Grade is A+")
#elif (marks > 70):
#     print ("Grade is A")
#elif (marks > 60):
#     print ("Grade is B")
#elif (marks < 50):
#     print ("Fail")

#---------- integer-----------------------------------------

#n=int(input("Enter last integer ="))
#for i in range (4,n+1):
#     if(int(i/5)-i/5==0):
#        print (i, "is divisible by 5")
#     elif (int(i/2)-i/2==0):
#        print (i, " is divisible by 2")

#-------Sirrrr -----------------------------------------
#n=int(input("Enter last integer ="))
#for i in range (1,n+1):
#     if(int(i/2)-i/2==0 or int(i/5)-i/5==0):
#        print (i)

#--------------Factorial------------------------------------------------------

#n=int(input("Enter an integer ="))
#product =1
#for i in range (1,n+1):
#    product=product*i
#    print ("The factorial of ",i,"is equal to =",product)
# -----------------------------------------------------------------

#import math
#n=int(input("Enter an integer ="))
#for i in range (1,n+1):
#    print ("The factorial of ",i,"is equal to =",math.factorial(i))
#-----------------------------------------------------------

#-------- Solve Quardatic Eqn--------------
#print ("Enter coefficient of Quadratic eq " )
#print ("=============================")
#a=float(input("Enter coefficent of x square ="))
#print ("=============================")
#b=float(input("Enter coefficent of x ="))
#print("==============================")
#c=float(input("Enter coefficient of x power zero ="))
#print ("=============================")
#d=-b/(2*a)
#e=(b**2-4*a*c)**0.5/(2*a)
#print("The roots are",d+e,d-e)

#-------- Another Method ------------------
#print("Enter coefficient of Quadratic eq " )
#print ("=============================")
#a=float(input("Enter coefficent of x square = "))
#print ("=============================")
#b=float(input("Enter coefficent of x ="))
#print("==============================")
#c=float(input("Enter coefficient of x power zero = "))
#print ("=============================")
#print ("The roots are",-b/(2*a)-(b**2-4*a*c)**0.5/(2*a),-b/(2*a)+(b**2-4*a*c)**0.5/(2*a))
#-------------------------------------------------------------------------------

#---------- Lecture #02------------------------
#import math
#r=float(input("enter a num = "))
#Area = math.pi*r**2
#print("Area is = ",Area)
#
#Volume = (4/3) * math.pi*r**3
#print("Volume is = ",Volume)
#
#base=float(input("Enter base of triangle = "))
#height=float(input("Enter height of triangle ="))
#Atri=((base*height)/2)
#print("Area of triangle  =" ,Atri)
#
#H=float(input("Enter Height in Feet = "))
#print(f"There are {30.48*height} Cm in {height} ft")

#--------------------Function-----------------------------------

#def fn(x):
#    return x**2
#print (fn(5))
#---------------------------------------
#def f(x,y):
#    return (x**2)+(y**2)
#print(f(3,2))
#---------------------------------
#def s(vi,a,t):
#    return vi*t +(.5*a*t**2)
#vi=float(input("enter initial value ="))
#a=float(input("enter a ="))
#t=float(input("enter t ="))
#print(s(vi,a,t))

#-------------While------------------------
#a=0
#while (a<=10):
#    a=int(input("enter a ="))
#    print(a)
#    a+=1
#--------------------------------------    
#for i in range(10):
#    y=i**5
#    print(y)

#-------------- Graphs-----------------
#import matplotlib.pyplot as plt
#x=[]
#y=[]
#for i in range (3,300):
#    y.append(i**2)
#    x.append(i)
#plt.plot(x,y)

#--------------Derivative-----------------------
#from sympy import diff
#from sympy import Symbol
#x=Symbol("x")
#y=x**2    
#z=y.diff(x)
#print(z)
#print(z.subs(x,4))

#---Chapter  # 05

#--------------------Bisection --------------------------
#def f(x):
#    return (x**3)-(6*x**2)-(261*x)+12
#a = 0
#while a == 0:
#    x1 = float(input("enter x1 =   "))
#    if (f(x1)== 0):
#        print("the root is x1 ")
#    x2 = float(input("enter x2 =  "))    
#    if (f(x2)== 0):
#        print("the root is x2 ")
#        
#    if (f(x1)*f(x2) < 0 ):
#         a += 1
#    else:
#        print("the root does not exist between x1 and x2")
#        
#xm = x1       
#epsilon = 0.0001
#print("x1 , x2 , xm , f(x1) , f(x2) , f(xm)")
#print('------------------------------------')
#
#while (abs(f(xm)> 0 )):
#    xm = (x1+x2/2)
#    if (f(xm)==0):
#        print("the root is x1 =",x1) 
#        break
#    print(round(x1,6),round(x2,6),round(xm,6),round(f(x1),6),round(f(x2),6),round(f(xm),6))
#    if (f(x1)*f(xm)<0):
#        x2 = xm
#    else:
#        x1 = xm
#print("--------------------")
#print("the root is xm =",xm)
        
# ------------- False Position Method ----------------
       
#def f(x):
#   return (x**3)-(6*x**2)-(261*x)+12
#a = 0
#while a == 0:
#    x1 = float(input("enter x1 = "))
#    if (f(x1)== 0):
#        print("the root is x1= ", x1)
#    x2 = float(input("enter x2 = "))    
#    if (f(x2)== 0):
#        print("the root is x2 =",x2)
#        
#    if (f(x1)*f(x2) < 0 ):
#         a += 1
#    else:
#        print("the root does not exist between x1 and x2")
#xm = x1
#epsilon = 0.0001
#print("x1 , x2 , x3 , f(x1) , f(x2) , f(x3)")
#print("------------------------------------")
#
#while (abs(f(xm)> 0 )):
#    xm = (x2 -(f(x2)*(x1-x2)) /(f(x1)-f(x2)) )
#    if (f(xm)==0):
#        print("the root is x1 =",x1)
#        break
#    print(round(x1,6),round(x2,6),round(xm,6),round(f(x1),6),round(f(x2),6),round(f(xm),6))
#    if (f(x1)*f(xm)<0):
#        x2 = xm
#    else:
#        x1 = xm
#print("--------------------")
#print("the root is xm =",xm) 
#

#chap #06    
#--------------------Secant Method -------------------
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

#-------------------Modified secant method----------------------
#
#import math
#def f(x):
#    return math.exp(-x)-x
#
#xi=float(input("Enter Initial Guess = "))
#e=0.000001
#dx=0.01
#xo=xi
#print("x1    f(x1)     xo     f(xo)")
#print("=====================================")
#
#while (abs(f(xo))>e):
#    D=xi+(dx)
#    xo=xi-((dx*f(xi))/(f(D)-f(xi)))
#    print (round(xi,5), round(f(xi),5), round(xo,5),round(f(xo),5))
#    xi=xo
#print ("==================")
#print ("the root is x0 ",xo)   


#---------------Newton Raphson Method----------------------------------

#import sympy
#from sympy import Symbol
#from sympy import diff
#
#r=Symbol('r')
#E=0.000001
#x1=float(input("enter initial guess = "))
#
#def f(x):
#    return x**3 +x**1 +12
#if (abs(f(x1))<E):
#    print("Root is =  ",x1)
#def fd(x):
#    z=diff(r**3 +r**1 +12) 
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
#
#    else:
#        x1=x2        

#--------------------Matrix Operation--------------------
#
#import numpy as np
#
#A=np.array([[1,0,0],[1,2,3],[1,2,4]])
#
#B=np.linalg.inv(A)
#
#C=np.matmul(A,B)
#
#D=np.transpose(A)
#
#print(A)
#
#print(B)
#
#print(C)
#
#print(D)
#
#print(np.linalg.det(A))
#
#print(np.linalg.det(B))

#-------------------Cramers Rule-------------------------
#import numpy as np
#A=np.array([[1,2,3],[3,5,6],[3,8,9]]) 
#B=np.array([[5,2,3],[4,5,6],[7,8,9]])
#     
#D=np.linalg.det(A)
#print("D=",D) 
#
#S=np.linalg.det(B)
#print("S=",S) 
#print("F=",D/S) 

#-----------------------#Generalize Cramers rule---------------

#import numpy as np 
#n=int(input("enter # of unknown="))
#col=[]
#for i in range (0,n):
#    row=[]
#    for j in range(0,n):
#        print("enter elements of A ",i,j)
#        x=int(input("enter # ="))
#        row.append(x)
#    col.append(row)
#
#A=np.matrix(col)
#print("A =",A)
#
#B=[]
#for k in range(0,n):
#    x1=int(input())
#    B.append(x1)
#B=np.array(B)
#print("B=",B)
#
#for w in range(0,n):
#    z=np.array(A)
#    for j in range (0,n):
#        z[j,i]=B[j]
#
#print("z=",z)
#print("t=",np.linalg.det(z)/np.linalg.det(A))

#--------------------Gauss Elimination Method-------------------------------

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
#input()
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
#-----------------Gauss Jordon--------------------
    
#import numpy as np
#n=int(input("enter # of unknown = "))
#col=[]
#for i in range (0,n):
#    row=[]
#    for j in range (0,n+1):
#        print("enter elements of A =",i,j)
#        x=int(input("coef ="))
#        row.append(x)
#    col.append(row)
#
#A=np.matrix(col)        
#print("A=",A)
#print("--------------------------")         
#
#input()
#for i in range(0,n):
#    for j in range(0,n):
#        A[i,:]=A[i,:]/A[i,i]
#        if (i!=j):
#            factor=A[j,i]/A[i,i]
#            for k in range (0,n+1):
#                A[j,k]=A[j,k]-factor*A[i,k]
#print(A)
#print("-------------------------")
#print(A[:,n])    
    
#------------------Least square fitting-------------------

#import matplotlib.pyplot as plt
#F= []
#a = []
#[sumx,sumy,sumxy,sumxx] = [0,0,0,0]
#n = int(input("enter total number of elements: "))
#for i in range(0,n):
#    F.append(int(input("enter elements of F:")))
#    a.append(int(input("enter elements of a:")))
#    sumx+=a[i]
#    sumy+= F[i]
#    sumxy+=a[i]*F[i]
#    sumxx+=a[i]*a[i]
#B = (n*sumxy - sumx*sumy)/ (n*sumxx - sumx**2)
#A =(sumy - B*sumx)/n
#print(A,B)

#Fnew = []
#for i in range(0,n):
#    Fnew.append(B*a[i]  + A)

#plt.polt(a,Fnew)
#plt.plot(a,F)

#----------------------Linear regression------------------------
#import numpy as np
#from numpy import log as ln
#import matplotlib.pyplot as plt
#
#x = []
#y = [] 
#X = []
#Y = []
#XY = []
#Xsquare = []
#n = int(input("enter # of elements = "))
#for i in range(0,n):
#    x.append(int(input("enter elements of x = ")))
#    y.append(int(input("enter elements of y = ")))
#    a  = ln(x)
#    b =  ln(y)
#X = np.sum(a)
#Y = np.sum(b)
#XY.append(np.sum(a*b)) 
#def power(lst,power):
#    return [i**power for i in lst]
#Xsquare.append(np.sum(power(a,2))) 
#B = (np.multiply(n,XY) - X*Y) / (np.multiply(n,Xsquare) - (X**2))
#A = (Y - B*X)/n
#a = np.e**A
#b = np.e**B
#print("b =", b)
#print("------------")
#print("a =", a)
#
#ynew = a*x **B
#plt.plot(ln(x),ynew)
#
#-------------linearization of a power equation---------

#import numpy as np
#from numpy import log as ln
#import matplotlib.pyplot as plt
#
#x = []
#y = [] 
#X = []
#Y = []
#XY = []
#Xsquare = []
#n = int(input("enter # of elements = "))
#for i in range(0,n):
#    x.append(int(input("enter xi = ")))
#    y.append(int(input("enter yi = ")))
#    a = ln(x)
#    b =  ln(y)
#X = np.sum(a)
#Y = np.sum(b)
#XY.append(np.sum(a*b))
#print("XY = " ,XY)
# 
#def power(lst,power):   #(sq list)
#    return [i**power for i in lst]
#Xsquare.append(np.sum(power(a,2))) 
#B = (np.multiply(n,XY) - X*Y) / (np.multiply(n,Xsquare) - (X**2))
#A = (Y - B*X)/n
#a = np.e**A
#print("B =",B ,"a =",a)
#
#ynew = a*x **B
#plt.plot(ln(x),ynew)

#------------------Remove and append -----------------------------------

#x=[2,5,1,9,10,2,6]
#y=[2,5,7,8,3,1,15]
#A=[]
#B=[]
#
#for i in range(0,len(x)):
#    A.append(max(x))
#    x.remove(max(x))
#    print(A,x)
#print("--------------------")
#
#for j in range(0,len(y)):
#    A.append(min(y))
#    y.remove(min(y))
#    print(A,y)

#---------------Smooth graph in Python----------------

#-you can increase data points by linspace and spline

#import numpy as np
#
#import matplotlib.pyplot as plt
#
#from scipy.interpolate import spline
#
#x=np.array([1,2,3,6])
#
#y=np.array([2,5,9,7])
#
#xx=np.linspace(1,6,50)
#
#yy=spline(x,y,xx)
#
#plt.plot(xx,yy,x,y,'o')  
    
    
    
    
    
    
    
    
    









