import csv

#days = 547 #até 21 de julho de 2021
#days = 783 #até 14 de março de 2022
days = 773 #3 de março de 2022, "lance do +1"
with open('time_series_covid19_deaths_global.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    countries = []
    number = []
    for i in range(days):
        number.append(0)
    for i in range(days):
        number[i] = [] #cada vetor number[i] será o numero de casos naquele dia i para determinado país
    for row in readCSV:
        country = row[1]
        countries.append(country)
        for i in range(days):
            number_ = row[i+4]
            number[i].append(number_)

#specific = [29,117,121,138,144,170,214,224,226]
#specific = [17,24,31,135,137,148,150,151,154,156,184,190,212,214,215,218,228,236,238,243,254,255,257,258,271,58,149,33,7,152,274,244,23,38,100,115,118,138,164,166,167,232,246,275,106]
specific = [233,32,34,100,199,237,18,155,257,216]
my_countries=[]

my_number = []
for i in range(days):
    my_number.append(0)
for i in range(days):
    my_number[i]=[]
        
for k in specific:
    my_countries.append(countries[k])
    for i in range(days):
        my_number[i].append(int(number[i][k]))
    
print (my_countries)

arquivo1 = open("Slovakia_deaths.dat","w")
arquivo2 = open("Brazil_deaths.dat","w")
arquivo3 = open("Bulgaria_deaths.dat","w")
arquivo4 = open("Croatia_deaths.dat","w")
arquivo5 = open("Netherlands_deaths.dat","w")
arquivo6 = open("South_Africa_deaths.dat","w")
arquivo7 = open("Austria_deaths.dat","w")
arquivo8 = open("Italy_deaths.dat","w")
arquivo9 = open("US_deaths.dat","w")
arquivo10 = open("Portugal_deaths.dat","w")

"""
arquivo1 = open("Austria_deaths.dat","w")
arquivo2 = open("Belgium_deaths.dat","w")
arquivo3 = open("Brazil_deaths.dat","w")
arquivo4 = open("Germany_deaths.dat","w")
arquivo5 = open("Greece_deaths.dat","w")
arquivo6 = open("India_deaths.dat","w")
arquivo7 = open("Iran_deaths.dat","w")
arquivo8 = open("Iraq_deaths.dat","w")
arquivo9 = open("Italy_deaths.dat","w")
arquivo10 = open("Japan_deaths.dat","w")
arquivo11 = open("Mexico_deaths.dat","w")
arquivo12 = open("Morocco_deaths.dat","w")
arquivo13 = open("Peru_deaths.dat","w")
arquivo14 = open("Poland_deaths.dat","w")
arquivo15 = open("Portugal_deaths.dat","w")
arquivo16 = open("Russia_deaths.dat","w")
arquivo17 = open("Serbia_deaths.dat","w")
arquivo18 = open("South_Africa_deaths.dat","w")
arquivo19 = open("Spain_deaths.dat","w")
arquivo20 = open("Sweden_deaths.dat","w")
arquivo21 = open("Turkey_deaths.dat","w")
arquivo22 = open("US_deaths.dat","w")
arquivo23 = open("Ukraine_deaths.dat","w")
arquivo24 = open("UAE_deaths.dat","w")
arquivo25 = open("Uruguay_deaths.dat","w")
arquivo26 = open("Chile_deaths.dat","w")
arquivo27 = open("Indonesia_deaths.dat","w")
arquivo28 = open("Bulgaria_deaths.dat","w")
arquivo29 = open("Argentina_deaths.dat","w")
arquivo30 = open("Ireland_deaths.dat","w")
arquivo31 = open("Venezuela_deaths.dat","w")
arquivo32 = open("Switzerland_deaths.dat","w")
arquivo33 = open("Belarus_deaths.dat","w")
arquivo34 = open("Cambodia_deaths.dat","w")
arquivo35 = open("Cuba_deaths.dat","w")
arquivo36 = open("Estonia_deaths.dat","w")
arquivo37 = open("Fiji_deaths.dat","w")
arquivo38 = open("Grenada_deaths.dat","w")
arquivo39 = open("Kyrgyzstan_deaths.dat","w")
arquivo40 = open("Latvia_deaths.dat","w")
arquivo41 = open("Lebanon_deaths.dat","w")
arquivo42 = open("Slovakia_deaths.dat","w")
arquivo43 = open("Taiwan_deaths.dat","w")
arquivo44 = open("Vietnam_deaths.dat","w")
arquivo45 = open("Denmark_deaths.dat","w")
"""
for i in range(days):
    arquivo1.write("%d \t %d \n" %(i+1,my_number[i][0]))
    arquivo2.write("%d \t %d \n" %(i+1,my_number[i][1]))
    arquivo3.write("%d \t %d \n" %(i+1,my_number[i][2]))
    arquivo4.write("%d \t %d \n" %(i+1,my_number[i][3]))
    arquivo5.write("%d \t %d \n" %(i+1,my_number[i][4]))
    arquivo6.write("%d \t %d \n" %(i+1,my_number[i][5]))
    arquivo7.write("%d \t %d \n" %(i+1,my_number[i][6]))
    arquivo8.write("%d \t %d \n" %(i+1,my_number[i][7]))
    arquivo9.write("%d \t %d \n" %(i+1,my_number[i][8]))
    arquivo10.write("%d \t %d \n" %(i+1,my_number[i][9]))
    """
    arquivo11.write("%d \t %d \n" %(i+1,my_number[i][10]))
    arquivo12.write("%d \t %d \n" %(i+1,my_number[i][11]))
    arquivo13.write("%d \t %d \n" %(i+1,my_number[i][12]))
    arquivo14.write("%d \t %d \n" %(i+1,my_number[i][13]))
    arquivo15.write("%d \t %d \n" %(i+1,my_number[i][14]))
    arquivo16.write("%d \t %d \n" %(i+1,my_number[i][15]))
    arquivo17.write("%d \t %d \n" %(i+1,my_number[i][16]))
    arquivo18.write("%d \t %d \n" %(i+1,my_number[i][17]))
    arquivo19.write("%d \t %d \n" %(i+1,my_number[i][18]))
    arquivo20.write("%d \t %d \n" %(i+1,my_number[i][19]))
    arquivo21.write("%d \t %d \n" %(i+1,my_number[i][20]))
    arquivo22.write("%d \t %d \n" %(i+1,my_number[i][21]))
    arquivo23.write("%d \t %d \n" %(i+1,my_number[i][22]))
    arquivo24.write("%d \t %d \n" %(i+1,my_number[i][23]))
    arquivo25.write("%d \t %d \n" %(i+1,my_number[i][24]))
    arquivo26.write("%d \t %d \n" %(i+1,my_number[i][25]))
    arquivo27.write("%d \t %d \n" %(i+1,my_number[i][26]))
    arquivo28.write("%d \t %d \n" %(i+1,my_number[i][27]))
    arquivo29.write("%d \t %d \n" %(i+1,my_number[i][28]))
    arquivo30.write("%d \t %d \n" %(i+1,my_number[i][29]))
    arquivo31.write("%d \t %d \n" %(i+1,my_number[i][30]))
    arquivo32.write("%d \t %d \n" %(i+1,my_number[i][31]))
    arquivo33.write("%d \t %d \n" %(i+1,my_number[i][32]))
    arquivo34.write("%d \t %d \n" %(i+1,my_number[i][33]))
    arquivo35.write("%d \t %d \n" %(i+1,my_number[i][34]))
    arquivo36.write("%d \t %d \n" %(i+1,my_number[i][35]))
    arquivo37.write("%d \t %d \n" %(i+1,my_number[i][36]))
    arquivo38.write("%d \t %d \n" %(i+1,my_number[i][37]))
    arquivo39.write("%d \t %d \n" %(i+1,my_number[i][38]))
    arquivo40.write("%d \t %d \n" %(i+1,my_number[i][39]))
    arquivo41.write("%d \t %d \n" %(i+1,my_number[i][40]))
    arquivo42.write("%d \t %d \n" %(i+1,my_number[i][41]))
    arquivo43.write("%d \t %d \n" %(i+1,my_number[i][42]))
    arquivo44.write("%d \t %d \n" %(i+1,my_number[i][43]))
    arquivo45.write("%d \t %d \n" %(i+1,my_number[i][44])) 
    """
arquivo1.close
arquivo2.close
arquivo3.close
arquivo4.close
arquivo5.close
arquivo6.close
arquivo7.close
arquivo8.close
arquivo9.close
arquivo10.close
"""
arquivo11.close
arquivo12.close
arquivo13.close
arquivo14.close
arquivo15.close
arquivo16.close
arquivo17.close
arquivo18.close
arquivo19.close
arquivo20.close
arquivo21.close
arquivo22.close
arquivo23.close
arquivo24.close
arquivo25.close
arquivo26.close
arquivo27.close
arquivo28.close
arquivo29.close
arquivo30.close
arquivo31.close
arquivo32.close
arquivo33.close
arquivo34.close
arquivo35.close
arquivo36.close
arquivo37.close
arquivo38.close
arquivo39.close
arquivo40.close
arquivo41.close
arquivo42.close
arquivo43.close
arquivo44.close
arquivo45.close
"""
