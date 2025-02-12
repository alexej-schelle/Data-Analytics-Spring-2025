################################################################################################################################################
#                                                                                                                                              #
#   Autor: Dr. A. Schelle (alexej.schelle.ext@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt    #
#                                                                                                                                              #
################################################################################################################################################

# PYTHON ROUTINE zur Darstellung von GAN-Netzwerken mit komplexen Zahlen in der Gaußschen Zahlenebene #

import os
import sys
import statistics
import math
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Definition eines Generators für das GAN-Modell

def Generator(length, input_key, train):

    output_key = [0]*length

    for j in range(0, length): # Generiere einen zufälligen binären Schlüssel zum Abgleich mit dem Referenzschlüssel

        output_key[j] = random.randint(0,1)

    return(output_key)

# Definition eines Diskriminators für das GAN-Modell

def Diskriminator(length, input_key, ideal_key):

    difference = [0]*length
    summe = 1.0

    for j in range(0, length):

        difference[j] = math.fabs(ideal_key[j] - input_key[j])*2**j # Berechne die Abweichung zwischen Referenzschlüssel und dynamisch generiertem Schlüssel
        summe = summe + 2**j

    for j in range(0, length):

        difference[j] = float(difference[j])/summe*100.0

    return(difference)

# Definition der Schaltungen für das GAN-Modell

def NOT(length, input_key): # Definiert ein NOT-Gate

    output_key = [0]*length
        
    for k in range(0, M):

        if (input_key[k] == 0): input_key[k] = 1        
        else: input_key[k] = 0
      
        output_key[k] = input_key[k]
        
    return output_key

def OR(length, input_key): # Definiert ein OR-Gate

    output_key = [0]*length
        
    for k in range(1, M):

        if (k == 8 or k == 6 or k == 4 or k == 2):
        
            if (input_key[k] == 0 and input_key[k-1] == 0): input_key[k-1] = 0
            if (input_key[k] == 0 and input_key[k-1] == 1): input_key[k-1] = 1        
            if (input_key[k] == 1 and input_key[k-1] == 0): input_key[k-1] = 1
            if (input_key[k] == 1 and input_key[k-1] == 1): input_key[k-1] = 1
              
    for k in range(0, M):

        output_key[k] = input_key[k]
        
    return output_key
        
def AND(length, input_key): # Definiert ein AND-Gate

    output_key = [0]*length
        
    for k in range(1, M):

        if (k == 8 or k == 6 or k == 4 or k == 2): 
  
            if (input_key[k] == 0 and input_key[k-1] == 0): input_key[k-1] = 0
            if (input_key[k] == 0 and input_key[k-1] == 1): input_key[k-1] = 0        
            if (input_key[k] == 1 and input_key[k-1] == 0): input_key[k-1] = 0
            if (input_key[k] == 1 and input_key[k-1] == 1): input_key[k-1] = 1
              
    for k in range(0, M):

        output_key[k] = input_key[k]
        
    return output_key

def XOR(length, input_key): # Definiert ein XOR-Gate ==> CNOT

    output_key = [0]*length
        
    for k in range(1, M):

        if (k == 8 or k == 6 or k == 4 or k == 2):

            if (input_key[k] == 0 and input_key[k-1] == 0): input_key[k-1] = 0
            if (input_key[k] == 0 and input_key[k-1] == 1): input_key[k-1] = 1        
            if (input_key[k] == 1 and input_key[k-1] == 0): input_key[k-1] = 1
            if (input_key[k] == 1 and input_key[k-1] == 1): input_key[k-1] = 0
              
    for k in range(0, M):

        output_key[k] = input_key[k]
        
    return output_key

def NAND(length, input_key): # Definiert ein NAND-Gate

    output_key = [0]*length
        
    for k in range(1, M):

        if (k == 8 or k == 6 or k == 4 or k == 2):
        
            if (input_key[k] == 0 and input_key[k-1] == 0): input_key[k-1] = 1
            if (input_key[k] == 0 and input_key[k-1] == 1): input_key[k-1] = 1        
            if (input_key[k] == 1 and input_key[k-1] == 0): input_key[k-1] = 1
            if (input_key[k] == 1 and input_key[k-1] == 1): input_key[k-1] = 0
              
    for k in range(0, M):

        output_key[k] = input_key[k]
        
    return output_key

def NOR(length, input_key): # Definiert ein NOR-Gate

    output_key = [0]*length
        
    for k in range(1, M):

        if (k == 8 or k == 6 or k == 4 or k == 2):
        
            if (input_key[k] == 0 and input_key[k-1] == 0): input_key[k-1] = 1
            if (input_key[k] == 0 and input_key[k-1] == 1): input_key[k-1] = 0        
            if (input_key[k] == 1 and input_key[k-1] == 0): input_key[k-1] = 0
            if (input_key[k] == 1 and input_key[k-1] == 1): input_key[k-1] = 0
              
    for k in range(0, M):

        output_key[k] = input_key[k]
                
    return output_key

# Definition eines GAN-Modells

def GAN(length, initial_key, reference_key):

    k = 0
    uncertainty = 10.00

    GAN_Key = [0]*length # Definiere den GAN-Schlüssel mit der Variable GAN_Key als Pythonliste mit der gleichen Anzahl von Elementen wie die anderen Keys 
        
    while(True):

        sum = 0.0
        initial_key = AND(length,OR(length, initial_key))

        if (k == 0): L = Generator(length, initial_key, reference_key) 

        if (k > 0): 

            GAN_Key = Generator(length, L, M)
            initial_key = GAN_Key
            L = AND(length,OR(length, GAN_Key))
        
        M = Diskriminator(length, L, reference_key)

        for j in range(0, length):

            sum = sum + math.fabs(M[j])

        k = k + 1

        if (sum < uncertainty): # Hier wird die Genauigkeit für das GAN festgelegt (Parameter uncertainty)

            return(initial_key)
            break

# Definition einer Funktion zur Generierung eines Referenz-Schlüssels

def GenerateReferenceKey(keysize):

    for j in range(0, keysize): # Generiere einen zufälligen binären Schlüssel als Referenzwert (entspricht externem und unbekanntem Referenzwert)

        R[j] = random.randint(0,1)
    
    return R

# Definition einer Funktion zur Generierung eines Schlüssel-Anfangswerts

def GenerateInitialKey(keysize):

    for j in range(0, keysize): # Generiere einen zufälligen binären Schlüssel als Startwert

        K[j] = random.randint(0,1)

    return K

S = 5000 # Anzahl der GAN-Konfigurationen (Anzahl der komplexen Zahlen in der Gaußschen Zahlenebene)   
M = 8 # Länge des Schlüssels in Einheiten von Bits

K = ['']*M # Definiere den Schlüssel K als Pythonliste mit der gleichen Anzahl von Elementen wie S     
R = ['']*M # Definiere den Schlüssel R als Pythonliste mit der gleichen Anzahl von Elementen wie S  

dkey = [] # Python-Liste zur Speicherung 
ARR_Keys = []
COMPLEX_Keys_Real = []
COMPLEX_Keys_Imag = []

dvalue = 0.0
fvalue = 0.0

print('Number of Key Elements : ', S)

for k in range(0, S):

    print('Generate Key Element Nr. ', k)

    K = GenerateInitialKey(M) # Generiere den ersten Schlüssel als Startwert für das GAN
    R = GenerateReferenceKey(M) # Generiere einen Referenzschlüssel (anfangs dengleichen Keywert)
    K = R # Lege dazu eine Anfangsbedingung mit K = R fest     

    ARR_Keys = ARR_Keys + R # Speichere jeweils den Referenzkey, der modulu Schaltung intrinsisch den Key K erzeugt

    key = GAN(M, K, R) # Modelliere ein GAN-Netzwerk zur Rekonstruktion eines der möglichen Eingangssignale (bisher unbekannt)

    dvalue = 0.0
    fvalue = 0.0

    for l in range(0, len(key)):

        dvalue = dvalue + 2**l*key[l]

    for l in range(0, len(key)):

        fvalue = fvalue + 2**l*R[l]
        
    dkey.append(int(dvalue))

    COMPLEX_Keys_Real.append(int(dvalue))
    COMPLEX_Keys_Imag.append(int(fvalue))

# Darstellung der komplexen Zahlen in der Gaußschen Zahlenebene

plt.figure(1)
plt.set_cmap("Blues")
plt.hist2d(COMPLEX_Keys_Real, COMPLEX_Keys_Imag, bins = 100)
plt.tick_params(axis='both', which='major', labelsize = 16)
plt.xlabel('Real Part', fontsize = 18)
plt.ylabel('Imaginary Part', fontsize = 18)
plt.savefig('fig_complex.png')