import numpy as np
from math import exp, sqrt, log, tanh
from scipy.optimize import curve_fit, minimize
from random import random, randint
from matplotlib import pyplot

arquivo = open("Brazil_dDdt_no_zeros.dat", "w")
arquivo2 = open("Brazil_dDdt_no_zeros_fit.dat", "w")
arquivo3  = open("Brazil_deaths_no_zeros.dat", "w")
arquivo4  = open("Brazil_deaths_no_zeros_fit.dat", "w")

data = np.loadtxt("Brazil_dDdt.dat")
N=len(data)

t_inicial = 0
for i in range(N):
    if data[i,1] !=0:
        t_inicial = i-1
        break
#t_inicial+=24
print (t_inicial)


t_mudanca =N-t_inicial#mudança da 1 pra 2 onda
x_1_onda = []
y_1_onda = []

for i in range(t_mudanca):
    x_1_onda.append(data[i+t_inicial,0]-t_inicial-1)
    y_1_onda.append(data[i+t_inicial,1])


def pathway_model_1_onda(x, C1,alfa1,q1,beta1,gamma1,C2,alfa2,q2,beta2,gamma2,rho,t0,C3,alfa3,q3,beta3,gamma3,rho2,t02):
    return (((C1+0.5*(C2-C1)*(1+np.tanh(0.5*rho*(x-t0)))+0.5*(C3-C2)*(1+np.tanh(0.5*rho2*(x-t02))))*
             ((x)**(alfa1+0.5*(alfa2-alfa1)*(1+np.tanh(0.5*rho*(x-t0)))+0.5*(alfa3-alfa2)*(1+np.tanh(0.5*rho2*(x-t02))))))
                          / ((1+((q1+0.5*(q2-q1)*(1+np.tanh(0.5*rho*(x-t0)))+0.5*(q3-q2)*(1+np.tanh(0.5*rho2*(x-t02))))-1)*
                              (beta1+0.5*(beta2-beta1)*(1+np.tanh(0.5*rho*(x-t0)))+0.5*(beta3-beta2)*(1+np.tanh(0.5*rho2*(x-t02))))*
                              ((x)**(gamma1+0.5*(gamma2-gamma1)*(1+np.tanh(0.5*rho*(x-t0)))+0.5*(gamma3-gamma2)*(1+np.tanh(0.5*rho2*(x-t02))))))**
                             (1/((q1+0.5*(q2-q1)*(1+np.tanh(0.5*rho*(x-t0)))+0.5*(q3-q2)*(1+np.tanh(0.5*rho2*(x-t02))))-1))))

##############
possible = []
for i in range(N-t_inicial):
        possible.append(i)

tam = len(possible)


################3
##x=#96#randint(0,N-t_inicial)
##y=#296#randint(x,N-t_inicial)

x =353# randint(100,400)
y = 648#randint(400,650)
print (x,y)
init_vals = [1e-3,4,1.4,1e-5,3,1e-3,4,1.4,1e-5,3,0.1,x,1e-3,4,1.4,1e-5,3,0.1,y]
#init_vals = [3.42e-18,14.7,1.14,3.84e-3,2.26,1.22e-15,17,1.08,0.0268,1.84,0.0499,x,1.33e-15,17.9,1.01,0.0204,1.32,0.0704,y]
bnds = [[0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0],[np.inf,np.inf,3,np.inf,np.inf,np.inf,np.inf,3,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,3,np.inf,np.inf,np.inf,np.inf]]
best_vals, covar = curve_fit(pathway_model_1_onda, x_1_onda, y_1_onda,p0=init_vals,bounds = bnds,  maxfev=5000000)
#u=[]
#for k in range(20):
#    u.append(sqrt(covar[k,k])/best_vals[k])
#for k in range(20):
#    if u[k]<1:
        #if k!=8 or k!=12 or k!=16:
#        bnds[0][k]=best_vals[k]-sqrt(covar[k,k])
#        bnds[1][k]=best_vals[k]+sqrt(covar[k,k])
temp = best_vals
#temp2 = bnds
for i in range(10):
    init_vals = temp
    #bnds = temp2
    print (i)
    best_vals, covar = curve_fit(pathway_model_1_onda, x_1_onda, y_1_onda,bounds = bnds, p0=init_vals,  maxfev=5000000)
    u0 =  sqrt(covar[0,0])/best_vals[0]
    u1 =  sqrt(covar[1,1])/best_vals[1]
    u2 = sqrt(covar[2,2])/best_vals[2]
    u3 =  sqrt(covar[3,3])/best_vals[3]
    u4 =  sqrt(covar[4,4])/best_vals[4]
    u5 =  sqrt(covar[5,5])/best_vals[5]
    u6 =  sqrt(covar[6,6])/best_vals[6]
    u7 = sqrt(covar[7,7])/best_vals[7]
    u8 =  sqrt(covar[8,8])/best_vals[8]
    u9 =  sqrt(covar[9,9])/best_vals[9]
    u10 =  sqrt(covar[10,10])/best_vals[10]
    u11 = sqrt(covar[11,11])/best_vals[11]
    u12 = sqrt(covar[12,12])/best_vals[12]
    u13 = sqrt(covar[13,13])/best_vals[13]
    u14 = sqrt(covar[14,14])/best_vals[14]
    u15 = sqrt(covar[15,15])/best_vals[15]
    u16 = sqrt(covar[16,16])/best_vals[16]
    u17 = sqrt(covar[17,17])/best_vals[17]
    u18 = sqrt(covar[18,18])/best_vals[18]
    
    tot=u0+u1+u2+u3+u4+u5+u6+u7+u8+u9+u10+u11+u12+u13+u14+u15+u16+u17+u18
    u=[]
    #for k in range(20):
    #    u.append(sqrt(covar[k,k])/best_vals[k])
    """
    else:
    """
    #print (u)
    if (u0 < 1 and u1 < 1 and u2 < 1 and u3 < 1 and u4< 1 and u5 < 1 and u6 < 1 and u7 < 1 and u8 < 1 and u9 < 1 and u10 < 1 and  u11 < 1 and  u12 < 1 and  u13 < 1 and  u14 < 1 and
        u15 < 1 and  u16 < 1 and  u17 < 1 and  u18 < 1):
        break
    else:
        temp = best_vals
        #for k in range(20):
        #    if u[k]<1:
                #if k!=8 or k!=12 or k!=16:
        #        temp2[0][k]=best_vals[k]-sqrt(covar[k,k])
        #        temp2[1][k]=best_vals[k]+sqrt(covar[k,k])     
    print (tot/19)

#print (best_vals)
"""
print (best_vals[0], np.sqrt(covar[0,0]))
print (best_vals[1], np.sqrt(covar[1,1]))
print (best_vals[2], np.sqrt(covar[2,2]))
print (best_vals[3], np.sqrt(covar[3,3]))
print (best_vals[4], np.sqrt(covar[4,4]))
"""
"""
for j in range(N-t_inicial):
    arquivo2.write("%d \t %d \n" %(x_[j], y_[j]))

arquivo2.close

for j in range(t_mudanca):
    arquivo3.write("%d \t %d \n" %(x_[j], pathway_model_1_onda(x_1_onda[j],best_vals[0],best_vals[1],best_vals[2],best_vals[3],best_vals[4])))
arquivo3.close
"""
######### agora incorporo a segunda onda
#def f(t,csi1, csi2, rho, t0):
    #return (csi1+0.5*(csi2-csi1)*(1+tanh(0.5*rho*(t-t0))))



"""
temp = [best_vals[0],best_vals[1],best_vals[2],best_vals[3],best_vals[4],best_vals[5],best_vals[6],best_vals[7],best_vals[8],best_vals[9],
        best_vals[10],best_vals[11]]

for i in range(1000):
    init_vals = temp
    print (i)
    best_vals , covar = curve_fit(pathway_model_2_onda, x_, y_, bounds = bnds, p0=init_vals,  maxfev=5000000)
    if (abs(best_vals[0]-temp[0])/temp[0] + abs(best_vals[1]-temp[1])/temp[1] + abs(best_vals[2]-temp[2])/temp[2] + abs(best_vals[3]-temp[3])/temp[3]
        + abs(best_vals[4]-temp[4])/temp[4] + abs(best_vals[5]-temp[5])/temp[5]+ abs(best_vals[6]-temp[6])/temp[6]+ abs(best_vals[7]-temp[7])/temp[7]+
        abs(best_vals[8]-temp[8])/temp[8]+ abs(best_vals[9]-temp[9])/temp[9]+ abs(best_vals[10]-temp[10])/temp[10]+ abs(best_vals[11]-temp[11])/temp[11]< 1.e-10 ):
        break
    else:
        temp = [best_vals[0],best_vals[1],best_vals[2],best_vals[3],best_vals[4],best_vals[5],best_vals[6],best_vals[7],best_vals[8],best_vals[9],
                best_vals[10],best_vals[11]]
"""
print ("C1 = "+str(best_vals[0])+"+/-"+str(sqrt(covar[0,0]))+"\n"+"alfa1 = "+str(best_vals[1])+"+/-"+str(sqrt(covar[1,1]))+"\n"+"q1 = "+str(best_vals[2])+"+/-"+str(sqrt(covar[2,2]))
        +"\n"+"beta1 = "+str(best_vals[3])+"+/-"+str(sqrt(covar[3,3]))+"\n"+"gamma1 = "+str(best_vals[4])+"+/-"+str(sqrt(covar[4,4]))+"\n"+"C2 = "+str(best_vals[5])+"+/-"+str(sqrt(covar[5,5]))
       +"\n"+"alfa2 = "+str(best_vals[6])+"+/-"+str(sqrt(covar[6,6]))+"\n"+"q2 = "+str(best_vals[7])+"+/-"+str(sqrt(covar[7,7]))+"\n"+"beta2 = "
       +str(best_vals[8])+"+/-"+str(sqrt(covar[8,8]))+"\n"+"gamma2 = "+str(best_vals[9])+"+/-"+str(sqrt(covar[9,9]))+"\n"+"rho = "+str(best_vals[10])
       +"+/-"+str(sqrt(covar[10,10]))+"\n"+"t0 = "+str(best_vals[11])+"+/-"+str(sqrt(covar[11,11]))+"\n"+"C3 = "+str(best_vals[12])+"+/-"+str(sqrt(covar[12,12]))+"\n"+"alfa3 = "+str(best_vals[13])+"+/-"+str(sqrt(covar[13,13]))
       +"\n"+"q3 = "+str(best_vals[14])+"+/-"+str(sqrt(covar[14,14]))+"\n"+"beta3 = "+str(best_vals[15])+"+/-"+str(sqrt(covar[15,15]))+"\n"+"gamma3 = "+str(best_vals[16])+"+/-"+str(sqrt(covar[16,16]))
       +"\n"+"rho2 = "+str(best_vals[17])+"+/-"+str(sqrt(covar[17,17]))+"\n"+"t02 = "+str(best_vals[18])+"+/-"+str(sqrt(covar[18,18])))

b=best_vals

#b = [6.3e-4,4.4,1.38,4.15e-5,3.083,0.85e-2,5.98,1.274,2.9e-5,3.15,0.081,252]#ugur
##########
"""
for k in range(N - t_inicial):
    arquivo.write("%d \t %d \n" %(x_[k], (((b[0]+0.5*(b[5]-b[0])*(1+tanh(0.5*b[10]*(x_[k]-b[11]))))*((x_[k])**(b[1]+0.5*(b[6]-b[1])*(1+tanh(0.5*b[10]*(x_[k]-b[11])))))) / ((1+((b[2]+0.5*(b[7]-b[2])*(1+tanh(0.5*b[10]*(x_[k]-b[11]))))-1)*(b[3]+0.5*(b[8]-b[3])*(1+tanh(0.5*b[10]*(x_[k]-b[11]))))*((x_[k])**(b[4]+0.5*(b[9]-b[4])*(1+tanh(0.5*b[10]*(x_[k]-b[11]))))))**(1/((b[2]+0.5*(b[7]-b[2])*(1+tanh(0.5*b[10]*(x_[k]-b[11]))))-1))))))
arquivo.close
"""
u0 =  sqrt(covar[0,0])/best_vals[0]
u1 =  sqrt(covar[1,1])/best_vals[1]
u2 = sqrt(covar[2,2])/best_vals[2]
u3 =  sqrt(covar[3,3])/best_vals[3]
u4 =  sqrt(covar[4,4])/best_vals[4]
u5 =  sqrt(covar[5,5])/best_vals[5]
u6 =  sqrt(covar[6,6])/best_vals[6]
u7 = sqrt(covar[7,7])/best_vals[7]
u8 =  sqrt(covar[8,8])/best_vals[8]
u9 =  sqrt(covar[9,9])/best_vals[9]
u10 =  sqrt(covar[10,10])/best_vals[10]
u11 = sqrt(covar[11,11])/best_vals[11]
u12 = sqrt(covar[12,12])/best_vals[12]
u13 = sqrt(covar[13,13])/best_vals[13]
u14 = sqrt(covar[14,14])/best_vals[14]
u15 = sqrt(covar[15,15])/best_vals[15]
u16 = sqrt(covar[16,16])/best_vals[16]
u17 = sqrt(covar[17,17])/best_vals[17]
u18 = sqrt(covar[18,18])/best_vals[18]
tot=u0+u1+u2+u3+u4+u5+u6+u7+u8+u9+u10+u11+u12+u13+u14+u15+u16+u17+u18

print (tot/19)


y_fit=[]
x_=x_1_onda
C1 = b[0]
alfa1=b[1]
q1=b[2]
beta1=b[3]
gamma1=b[4]
C2=b[5]
alfa2=b[6]
q2=b[7]
beta2=b[8]
gamma2=b[9]
rho=b[10]
t0=b[11]
C3=b[12]
alfa3=b[13]
q3=b[14]
beta3=b[15]
gamma3=b[16]
rho2=b[17]
t02=b[18]
for k in range(t_mudanca):
    y_fit.append(((C1+0.5*(C2-C1)*(1+np.tanh(0.5*rho*(x_[k]-t0)))+0.5*(C3-C2)*(1+np.tanh(0.5*rho2*(x_[k]-t02))))
                  *((x_[k])**(alfa1+0.5*(alfa2-alfa1)*(1+np.tanh(0.5*rho*(x_[k]-t0)))+0.5*(alfa3-alfa2)*(1+np.tanh(0.5*rho2*(x_[k]-t02))))))
                          / ((1+((q1+0.5*(q2-q1)*(1+np.tanh(0.5*rho*(x_[k]-t0)))+0.5*(q3-q2)*(1+np.tanh(0.5*rho2*(x_[k]-t02))))-1)*(beta1+0.5*(beta2-beta1)*(1+np.tanh(0.5*rho*(x_[k]-t0)))+0.5*(beta3-beta2)*(1+np.tanh(0.5*rho2*(x_[k]-t02))))*
                              ((x_[k])**(gamma1+0.5*(gamma2-gamma1)*(1+np.tanh(0.5*rho*(x_[k]-t0)))+0.5*(gamma3-gamma2)*(1+np.tanh(0.5*rho2*(x_[k]-t02))))))**
                             (1/((q1+0.5*(q2-q1)*(1+np.tanh(0.5*rho*(x_[k]-t0)))+0.5*(q3-q2)*(1+np.tanh(0.5*rho2*(x_[k]-t02))))-1))))

###############
print ("cond.2=>",gamma3-(q3-1)*(alfa3+1))
#################
fig=pyplot.figure(figsize=(8, 6))
pyplot.scatter(x_1_onda,y_1_onda,c='lime')
pyplot.plot(x_1_onda,y_fit, 'k',linewidth=2)
pyplot.xlabel("days since first death")
pyplot.ylabel("daily number of deaths")
pyplot.title("Brazil")
pyplot.show()
pyplot.savefig("Brazil_dDdt.eps",bbox_inches='tight')

#################################################


data2 = np.loadtxt("Brazil_deaths.dat")
N2=len(data2)

t_inicial2 = 0
for i in range(N2):
    if data2[i,1] !=0:
        t_inicial2 = i-1
        break



t_mudanca2 =N2-t_inicial2#mudança da 1 pra 2 onda
x_1_onda2 = []
y_1_onda2 = []

for i in range(t_mudanca2):
    x_1_onda2.append(data2[i+t_inicial2,0]-t_inicial2-1)
    y_1_onda2.append(data2[i+t_inicial2,1])

y_fit2 =[]
aux= 0
for k in range(t_mudanca2):
    y_fit2.append(aux)
    aux=aux+y_fit[k]

fig=pyplot.figure(figsize=(8, 6))
pyplot.scatter(x_1_onda2,y_1_onda2,c='lime')
pyplot.plot(x_1_onda2,y_fit2, 'k',linewidth=2)
pyplot.xlabel("days since first death")
pyplot.ylabel("cumulative number of deaths")
pyplot.title("Brazil")
#pyplot.show()
pyplot.savefig("Brazil_deaths.eps",bbox_inches='tight')

for m in range(t_mudanca):
    arquivo.write("%d \t %d \n" %(x_1_onda[m], y_1_onda[m]))
    arquivo2.write("%d \t %f \n" %(x_1_onda[m], y_fit[m]))
    arquivo3.write("%d \t %d \n" %(x_1_onda2[m], y_1_onda2[m]))
    arquivo4.write("%d \t %f \n" %(x_1_onda2[m], y_fit2[m]))
arquivo.close
arquivo2.close
arquivo3.close
arquivo4.close
