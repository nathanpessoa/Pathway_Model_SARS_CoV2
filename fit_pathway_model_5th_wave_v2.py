import numpy as np
from math import exp, sqrt, log, tanh
from scipy.optimize import curve_fit, minimize
from random import random, randint
from matplotlib import pyplot

arquivo = open("Austria_dDdt_no_zeros.dat", "w")
arquivo2 = open("Austria_dDdt_no_zeros_fit.dat", "w")
arquivo3  = open("Austria_deaths_no_zeros.dat", "w")
arquivo4  = open("Austria_deaths_no_zeros_fit.dat", "w")

data = np.loadtxt("Austria_dDdt.dat")
N=len(data)

t_inicial = 0
for i in range(N):
    if data[i,1] !=0:
        t_inicial = i-1
        break

print (t_inicial)


t_mudanca =N-t_inicial#mudança da 1 pra 2 onda
x_1_onda = []
y_1_onda = []

for i in range(t_mudanca):
    x_1_onda.append(data[i+t_inicial,0]-t_inicial-1)
    y_1_onda.append(data[i+t_inicial,1])


def pathway_model_1_onda(x, C1,alfa1,q1,beta1,gamma1,C2,alfa2,q2,beta2,gamma2,rho,t0,C3,alfa3,q3,beta3,gamma3,rho2,t02,C4,alfa4,q4,beta4,gamma4,rho3,t03,C5,alfa5,q5,beta5,gamma5,rho4,t04):
    return (((C1+0.5*(C2-C1)*(1+np.tanh(0.5*rho*(x-t0)))+0.5*(C3-C2)*(1+np.tanh(0.5*rho2*(x-t02)))+0.5*(C4-C3)*(1+np.tanh(0.5*rho3*(x-t03)))+0.5*(C5-C4)*(1+np.tanh(0.5*rho4*(x-t04))))*
             ((x)**(alfa1+0.5*(alfa2-alfa1)*(1+np.tanh(0.5*rho*(x-t0)))+0.5*(alfa3-alfa2)*(1+np.tanh(0.5*rho2*(x-t02)))+0.5*(alfa4-alfa3)*(1+np.tanh(0.5*rho3*(x-t03)))+0.5*(alfa5-alfa4)*(1+np.tanh(0.5*rho4*(x-t04))))))
                          / ((1+((q1+0.5*(q2-q1)*(1+np.tanh(0.5*rho*(x-t0)))+0.5*(q3-q2)*(1+np.tanh(0.5*rho2*(x-t02)))+0.5*(q4-q3)*(1+np.tanh(0.5*rho3*(x-t03)))+0.5*(q5-q4)*(1+np.tanh(0.5*rho4*(x-t04))))-1)*
                              (beta1+0.5*(beta2-beta1)*(1+np.tanh(0.5*rho*(x-t0)))+0.5*(beta3-beta2)*(1+np.tanh(0.5*rho2*(x-t02)))+0.5*(beta4-beta3)*(1+np.tanh(0.5*rho3*(x-t03)))+0.5*(beta5-beta4)*(1+np.tanh(0.5*rho4*(x-t04))))*
                              ((x)**(gamma1+0.5*(gamma2-gamma1)*(1+np.tanh(0.5*rho*(x-t0)))+0.5*(gamma3-gamma2)*(1+np.tanh(0.5*rho2*(x-t02)))+0.5*(gamma4-gamma3)*(1+np.tanh(0.5*rho3*(x-t03)))+0.5*(gamma5-gamma4)*(1+np.tanh(0.5*rho4*(x-t04))))))**
                             (1/((q1+0.5*(q2-q1)*(1+np.tanh(0.5*rho*(x-t0)))+0.5*(q3-q2)*(1+np.tanh(0.5*rho2*(x-t02)))+0.5*(q4-q3)*(1+np.tanh(0.5*rho3*(x-t03)))+0.5*(q5-q4)*(1+np.tanh(0.5*rho4*(x-t04))))-1))))
    #init_vals = [6.3e-4,4.4,1.38,4.15e-5,3.083]
"""
init_vals = [0.00088505021,4.3716185, 1.4392006,0.00021081465,3.0378692,0.0014136433, 4.2511851 ,1.4311444, 1.9040851e-05, 3.1674755, 
             0.071310313,124,0.0018772046,4.2499947 ,1.2844321,3.4684831e-06,3.0586453, 0.0699816862, 221 ,0.0033414346, 3.9562118 ,1.0947838,
             2.4347573e-06 , 2.6700888 , 0.085796792, 366]
print (init_vals)
bnds = (0.00088505020,4.3716184,1.4392005,0.00021081464,3.0378691,0.0014136432,4.2511850,1.4311443,1.90408510e-05,3.1674754,
        0.071310312,124-0.00001,0.001877203,4.2499946,1.2844320,3.4684830e-06,3.0586452,0.069981685,221-0.00001,0.0033414345,3.9562117,1.0947837,
        2.4347572e-06,2.6700887,0.085796791,366-0.00001),(0.00088505022,4.3716186,1.4392007,0.00021081466,3.0378693,0.0014136434,4.2511852,1.4311445,1.90408512e-05,3.1674756,
         0.071310314,124+0.00001,0.001877205,4.2499948,1.2844322,3.4684832e-06,3.0586454,0.069981687,221+0.00001,0.0033414346,3.95621189,1.0947839,
         2.4347574e-06,2.6700889,0.085796793,366+0.00001)
"""

##############
possible = []
for i in range(N-t_inicial):
        possible.append(i)

tam = len(possible)


################
x = 174#randint(30,280)#174#
y = 358#randint(280,400)#358#
z = 512#randint(400,630)#512
w = 692#randint(630,tam-1)#692#
print (x,y,z,w)
init_vals = [1e-3,4,1.4,1e-5,3,1e-3,4,1.4,1e-5,3,0.1,x,1e-3,4,1.4,1e-5,3,0.1,y,1e-3,4,1.4,1e-5,3,0.1,z,1e-3,4,1.4,1e-5,3,0.1,w]
#init_vals = [9.63e-17,15.1,1.27,1.21,4.63,
 #            1.07e-7,14.2,3,0.00103,22.4,0.230,x,
  #           7.28e-10,15,1.16,0.908,2.06,0.167,y,
   #          4.45e-9,17.6,1,11.4,0.243,0.0482,z,
    #         4.46e-13,16.2,1.61,7.66e-3,7.78,0.0746,w]
bnds = [[0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0],[np.inf,np.inf,3,np.inf,np.inf,np.inf,np.inf,3,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,3,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,3,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,3,np.inf,np.inf,np.inf,np.inf]]

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
    u19 = sqrt(covar[19,19])/best_vals[19]
    u20 = sqrt(covar[20,20])/best_vals[20]
    u21 = sqrt(covar[21,21])/best_vals[21]
    u22 = sqrt(covar[22,22])/best_vals[22]
    u23 = sqrt(covar[23,23])/best_vals[23]
    u24 = sqrt(covar[24,24])/best_vals[24]
    u25 = sqrt(covar[25,25])/best_vals[25]
    u26 = sqrt(covar[26,26])/best_vals[26]
    u27 = sqrt(covar[27,27])/best_vals[27]
    u28 = sqrt(covar[28,28])/best_vals[28]
    u29 = sqrt(covar[29,29])/best_vals[29]
    u30 = sqrt(covar[30,30])/best_vals[30]
    u31 = sqrt(covar[31,31])/best_vals[31]
    u32 = sqrt(covar[32,32])/best_vals[32]
    
    tot=u0+u1+u2+u3+u4+u5+u6+u7+u8+u9+u10+u11+u12+u13+u14+u15+u16+u17+u18+u19+u20+u21+u22+u23+u24+u25+u26+u27+u28+u29+u30+u31+u32
    u=[]
    #for k in range(20):
    #    u.append(sqrt(covar[k,k])/best_vals[k])
    """
    else:
    """
    #print (u)
    if (u0 < 1 and u1 < 1 and u2 < 1 and u3 < 1 and u4< 1 and u5 < 1 and u6 < 1 and u7 < 1 and u8 < 1 and u9 < 1 and u10 < 1 and  u11 < 1 and  u12 < 1 and  u13 < 1 and  u14 < 1 and
        u15 < 1 and  u16 < 1 and  u17 < 1 and  u18 < 1 and  u19 < 1 and  u20 < 1 and  u21 < 1 and  u22 < 1 and  u23 < 1 and  u24 < 1 and  u25 < 1 and  u26 < 1 and  u27 < 1 and
            u28 < 1 and  u29 < 1 and  u30 < 1 and  u31 < 1 and  u32 < 1):
        break
    else:
        temp = best_vals
        #for k in range(20):
        #    if u[k]<1:
                #if k!=8 or k!=12 or k!=16:
        #        temp2[0][k]=best_vals[k]-sqrt(covar[k,k])
        #        temp2[1][k]=best_vals[k]+sqrt(covar[k,k])     
    print (tot/33)

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
       +"\n"+"rho2 = "+str(best_vals[17])+"+/-"+str(sqrt(covar[17,17]))+"\n"+"t02 = "+str(best_vals[18])+"+/-"+str(sqrt(covar[18,18]))+"\n"+"C4 = "+str(best_vals[19])+"+/-"+str(sqrt(covar[19,19]))+"\n"+"alfa4 = "+str(best_vals[20])+"+/-"+str(sqrt(covar[20,20]))
       +"\n"+"q4 = "+str(best_vals[21])+"+/-"+str(sqrt(covar[21,21]))+"\n"+"beta4 = "+str(best_vals[22])+"+/-"+str(sqrt(covar[22,22]))+"\n"+"gamma4 = "+str(best_vals[23])+"+/-"+str(sqrt(covar[23,23]))
       +"\n"+"rho3 = "+str(best_vals[24])+"+/-"+str(sqrt(covar[24,24]))+"\n"+"t03 = "+str(best_vals[25])+"+/-"+str(sqrt(covar[25,25]))+"\n"+"C5 = "+str(best_vals[26])+"+/-"+str(sqrt(covar[26,26]))+"\n"+"alfa5 = "+str(best_vals[27])+"+/-"+str(sqrt(covar[27,27]))
       +"\n"+"q5 = "+str(best_vals[28])+"+/-"+str(sqrt(covar[28,28]))+"\n"+"beta5 = "+str(best_vals[29])+"+/-"+str(sqrt(covar[29,29]))+"\n"+"gamma5 = "+str(best_vals[30])+"+/-"+str(sqrt(covar[30,30]))
       +"\n"+"rho4 = "+str(best_vals[31])+"+/-"+str(sqrt(covar[31,31]))+"\n"+"t04 = "+str(best_vals[32])+"+/-"+str(sqrt(covar[32,32])))

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
u19 = sqrt(covar[19,19])/best_vals[19]
u20 = sqrt(covar[20,20])/best_vals[20]
u21 = sqrt(covar[21,21])/best_vals[21]
u22 = sqrt(covar[22,22])/best_vals[22]
u23 = sqrt(covar[23,23])/best_vals[23]
u24 = sqrt(covar[24,24])/best_vals[24]
u25 = sqrt(covar[25,25])/best_vals[25]
u26 = sqrt(covar[26,26])/best_vals[26]
u27 = sqrt(covar[27,27])/best_vals[27]
u28 = sqrt(covar[28,28])/best_vals[28]
u29 = sqrt(covar[29,29])/best_vals[29]
u30 = sqrt(covar[30,30])/best_vals[30]
u31 = sqrt(covar[31,31])/best_vals[31]
u32 = sqrt(covar[32,32])/best_vals[32]
tot=u0+u1+u2+u3+u4+u5+u6+u7+u8+u9+u10+u11+u12+u13+u14+u15+u16+u17+u18+u19+u20+u21+u22+u23+u24+u25+u26+u27+u28+u29+u30+u31+u32

print (tot/33)


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
C4=b[19]
alfa4=b[20]
q4=b[21]
beta4=b[22]
gamma4=b[23]
rho3=b[24]
t03=b[25]
C5=b[26]
alfa5=b[27]
q5=b[28]
beta5=b[29]
gamma5=b[30]
rho4=b[31]
t04=b[32]

###############
print ("cond.2=>",gamma5-(q5-1)*(alfa5+1))
#################
for k in range(t_mudanca):
    y_fit.append(((C1+0.5*(C2-C1)*(1+np.tanh(0.5*rho*(x_[k]-t0)))+0.5*(C3-C2)*(1+np.tanh(0.5*rho2*(x_[k]-t02)))+0.5*(C4-C3)*(1+np.tanh(0.5*rho3*(x_[k]-t03)))+0.5*(C5-C4)*(1+np.tanh(0.5*rho4*(x_[k]-t04))))
                  *((x_[k])**(alfa1+0.5*(alfa2-alfa1)*(1+np.tanh(0.5*rho*(x_[k]-t0)))+0.5*(alfa3-alfa2)*(1+np.tanh(0.5*rho2*(x_[k]-t02)))+0.5*(alfa4-alfa3)*(1+np.tanh(0.5*rho3*(x_[k]-t03)))+0.5*(alfa5-alfa4)*(1+np.tanh(0.5*rho4*(x_[k]-t04))))))
                          / ((1+((q1+0.5*(q2-q1)*(1+np.tanh(0.5*rho*(x_[k]-t0)))+0.5*(q3-q2)*(1+np.tanh(0.5*rho2*(x_[k]-t02)))+0.5*(q4-q3)*(1+np.tanh(0.5*rho3*(x_[k]-t03)))+0.5*(q5-q4)*(1+np.tanh(0.5*rho4*(x_[k]-t04))))-1)*(beta1+0.5*(beta2-beta1)*(1+np.tanh(0.5*rho*(x_[k]-t0)))+0.5*(beta3-beta2)*(1+np.tanh(0.5*rho2*(x_[k]-t02)))+0.5*(beta4-beta3)*(1+np.tanh(0.5*rho3*(x_[k]-t03)))+0.5*(beta5-beta4)*(1+np.tanh(0.5*rho4*(x_[k]-t04))))*
                              ((x_[k])**(gamma1+0.5*(gamma2-gamma1)*(1+np.tanh(0.5*rho*(x_[k]-t0)))+0.5*(gamma3-gamma2)*(1+np.tanh(0.5*rho2*(x_[k]-t02)))+0.5*(gamma4-gamma3)*(1+np.tanh(0.5*rho3*(x_[k]-t03)))+0.5*(gamma5-gamma4)*(1+np.tanh(0.5*rho4*(x_[k]-t04))))))**
                             (1/((q1+0.5*(q2-q1)*(1+np.tanh(0.5*rho*(x_[k]-t0)))+0.5*(q3-q2)*(1+np.tanh(0.5*rho2*(x_[k]-t02)))+0.5*(q4-q3)*(1+np.tanh(0.5*rho3*(x_[k]-t03)))+0.5*(q5-q4)*(1+np.tanh(0.5*rho4*(x_[k]-t04))))-1))))


fig=pyplot.figure(figsize=(8, 6))
pyplot.scatter(x_1_onda,y_1_onda,c='lime')
pyplot.plot(x_1_onda,y_fit, 'k',linewidth=2)
pyplot.xlabel("days since first death")
pyplot.ylabel("daily number of deaths")
pyplot.title("Austria")
pyplot.show()
pyplot.savefig("Austria_dDdt.eps", bbox_inches='tight')

#################################################


data2 = np.loadtxt("Austria_deaths.dat")
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
pyplot.title("Austria")
#pyplot.show()
pyplot.savefig("Austria_deaths.eps", bbox_inches='tight')

for m in range(t_mudanca):
    arquivo.write("%d \t %d \n" %(x_1_onda[m], y_1_onda[m]))
    arquivo2.write("%d \t %f \n" %(x_1_onda[m], y_fit[m]))
    arquivo3.write("%d \t %d \n" %(x_1_onda2[m], y_1_onda2[m]))
    arquivo4.write("%d \t %f \n" %(x_1_onda2[m], y_fit2[m]))
arquivo.close
arquivo2.close
arquivo3.close
arquivo4.close

print (C1,',',alfa1,',',q1,',',beta1,',',gamma1,',',C2,',',alfa2,',',q2,',',beta2,',',gamma2,',',rho,',','x',',',C3,',',alfa3,',',q3,',',beta3,',',gamma3,',',rho2,',','y'
,',',C4,',',alfa4,',',q4,',',beta4,',',gamma4,',',rho3,',','z',',',C5,',',alfa5,',',q5,',',beta5,',',gamma5,',',rho4,',','w')

