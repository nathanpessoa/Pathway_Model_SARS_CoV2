import numpy as np

data = np.loadtxt("Slovakia_deaths.dat")
#data2 = np.loadtxt("Italy_recovered_cases.dat")
#data2=np.loadtxt("UK_active_cases.dat")
arquivo = open("Slovakia_dDdt.dat","w")
#arquivo2 = open("Italy_dRdt.dat","w")
N=len(data)
for i in range(N-1):
    arquivo.write("%d \t %d \n" %(data[i,0],data[i+1,1]-data[i,1]))
    #arquivo2.write("%d \t %d \n" %(data2[i,0],data2[i+1,1]-data2[i,1]))
arquivo.close
#arquivo2.close
