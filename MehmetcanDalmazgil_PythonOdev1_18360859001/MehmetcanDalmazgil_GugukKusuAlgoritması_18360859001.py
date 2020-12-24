#################### GUGUK KUŞU ALGORİTMASI #####################

################## TEST FONKSİYONLARI ##################
# Beale Fonksiyonu (fmin(3,0.5) = 0)
# Ackley Fonksiyonu (fmin(0,0) = 0)
# Goldstein–Price Fonksiyonu (fmin(0,-1) = 3)
# Levi Fonksiyonu (fmin(1,1) = 0)

from random import *
import numpy as np
import math
from math import *
import matplotlib.pyplot as plt

class Guguk_Kusu_Algoritmasi():
    def __init__(self): # Kullanicidan populasyon ve parametreler hakkinda bilgi alinan kisim
        
        # Farkli yuvalarin/cozumlerin sayisi
        n = 20

        # Cozumlerin kac boyutlu vektorler oldugu
        dimension = 2

        # Guguk kusu yumurtalarinin kesfedilme orani
        pa = 1

        # Test fonksiyonunun secilecegi kisim
        print("\n")
        print("1 --> Beale Fonksiyonu (fmin(3,0.5) = 0)")
        print("2 --> Ackley Fonksiyonu (fmin(0,0) = 0)")
        print("3 --> Goldstein–Price Fonksiyonu (fmin(0,-1) = 3)")
        print("4 --> Levi Fonksiyonu (fmin(1,1) = 0)")
        print("\n")
        kontrol = True
        self.sec = 0
        while(kontrol):
            self.sec = int(input("Kullanmak istediginiz test fonksiyonunun numarasını giriniz : "))
            if(self.sec != 1 and self.sec != 2 and self.sec != 3 and self.sec != 4):
                print("Gecerli bir değer giriniz.")
            else:
                kontrol = False
        print("\n")
        
        if(self.sec == 1):
            print("Beale fonksiyonunun dogru değerlere en az adımda yaklasması için x ve y degerlerinin -4.5 ile 4.5 arasında olması gerekmektedir.  ")
            # Bir cozumdeki/yuvadaki bir bilesenin alabilecegi minimumu deger
            min_deger=float(input("Minimum Deger : "))
            # Bir cozumdeki/yuvadaki bir bilesenin alabilecegi maximum deger
            max_deger=float(input("Maximum Deger : "))
        if(self.sec == 2):
            print("Ackley fonksiyonunun dogru değerlere en az adımda yaklasması için x ve y degerlerinin -5 ile 5 arasında olması gerekmektedir.  ")
            # Bir cozumdeki/yuvadaki bir bilesenin alabilecegi minimumu deger
            min_deger=float(input("Minimum Deger : "))
            # Bir cozumdeki/yuvadaki bir bilesenin alabilecegi maximum deger
            max_deger=float(input("Maximum Deger : "))
        if(self.sec == 3):
            print("Goldstein–Price fonksiyonunun dogru en az adımda değerlere yaklasması için x ve y degerlerinin -2 ile 2 arasında olması gerekmektedir.  ")
            # Bir cozumdeki/yuvadaki bir bilesenin alabilecegi minimumu deger
            min_deger=float(input("Minimum Deger : "))
            # Bir cozumdeki/yuvadaki bir bilesenin alabilecegi maximum deger
            max_deger=float(input("Maximum Deger : "))
        if(self.sec == 4):
            print("Levi fonksiyonunun dogru değerlere en az adımda yaklasması için x ve y degerlerinin -10 ile 10 arasında olması gerekmektedir.  ")
            # Bir cozumdeki/yuvadaki bir bilesenin alabilecegi minimumu deger
            min_deger=float(input("Minimum Deger : "))
            # Bir cozumdeki/yuvadaki bir bilesenin alabilecegi maximum deger
            max_deger=float(input("Maximum Deger : "))
        print("\n")

        # Iterasyon sayisi (Daha iyi sonuclar almak icin artirilmalidir.)
        print("İterasyon sayisi daha iyi sonuc icin yuksek verilmelidir.")
        iteration=int(input("İterasyon Sayısı : "))
        print("\n")

        # Random olarak olusturulan ilk populasyon/cozum degerleri
        self.nests = np.array([[uniform(min_deger, max_deger) for k in range(dimension)] for i in range(n)])
        
        # Her bir cozum icin fonksiyondaki degerini bul
        self.fitness=(10**10)*np.ones(n)

        # Su anki en iyi cozumu al
        self.fmin,self.bestnest,self.nests,self.fitness=self.En_Iyi_Cozum(self.nests,self.nests,self.fitness)

        #Iterasyonlara basla
        self.count=0

        best_x = []
        best_y = []
        best_tablo = []

        # Iterasyon sayisi kadar donen dongu 
        while(self.count<iteration):

            # Yeni cozumler uret, mevcut en iyiyi tut
            self.new_nest=self.Guguk_Kusu_Bul(self.nests,self.bestnest,min_deger,max_deger)
            self.fnew,self.best,self.nests,self.fitness=self.En_Iyi_Cozum(self.nests,self.new_nest,self.fitness)

            # Sayaci guncelle
            self.count=self.count+n; 

            # Kesif ve randomization
            new_nest=self.Bos_Yuva(self.nests,min_deger,max_deger,pa)

            # Su anki cozum kumelerindeki hesaplamalari yap
            self.fnew,self.best,self.nests,self.fitness=self.En_Iyi_Cozum(self.nests,self.new_nest,self.fitness)

            # Sayaci guncelle
            self.count=self.count+n

            # Simdiye kadarki en iyi cozumu bul
            if self.fnew<self.fmin:
                self.fmin=self.fnew
                self.bestnest=self.best

            # Her iterasyonda buldugu en iyi cozumu gosteren grafigi cizen kod parcasi
            plt.grid()
            x = self.best[0]
            y = self.best[1]
            plt.plot(x, y, 'D', color="b")

            # Yapilan denemelerde 100000 iterasyon sonucunda cozum degerleri ile fonksiyon sonucunun beklenen degerlere minimize edildigi gorulmustur. 
            # Iterasyon sayisi cok buyuk oldugundan gorsellestirme icin her 1000 iterasyonu temsili olarak 1 adet cozum degeri ve fonksiyon sonucu alinmistir.  
            if(self.count%1000 == 0):
                best_x.append(self.best[0])
                best_y.append(self.best[1])
                best_tablo.append(self.fmin)
            
            # Her iterasyon sonucunda olusturulan x,y degerler
            print(f"{self.count}. İterasyon Sonuc ==> f({self.bestnest[0]},{self.bestnest[1]}) = {self.fmin}")
            print("---------------------------------------------------------------------------------------------")
         
        # En iyi cozumu ve fonksiyonda aldigi degeri ekrana yaz
        print(f"En İyi Çözüm = {self.bestnest} ")
        print(f"fmin = {self.fmin}")

        # Her iterasyonda buldugu en iyi cozumu ve fonksiyon sonucunu gosteren grafigi cizen kod parcasi
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(best_tablo,'r.-')
        ax1.legend(['f_min'])
        ax2 = fig.add_subplot(212)
        ax2.plot(best_x,'b.-')
        ax2.plot(best_y,'g--')
        ax2.legend(['x','y'])

        plt.show()

    # Yeni sonuclar üreterek bazi yuvalari/cozumleri bunlarla degistir. 
    # pa olasilikla en kotu cozumler kesfedilip yenileriyle degistirilecek.
    def Bos_Yuva(self,nest, min_deger,max_deger,pa):
        n=len(nest)
        dimension=len(nest[0])

        # Cozumun kesfedilip kesfedilmedigini tutan durum vektoru 
        K=np.array([np.random.random([len(nest),dimension])<pa],dtype=int)

        # Bir guguk kusunun yumurtasi eger ev sahibi yumurtaya cok benziyorsa o zaman kesfedilmesi dusuk bir olasiliktir.
        # Bu yuzden fitness cozumler arasindaki farkla iliskili olmali
        stepsize=np.multiply(np.subtract(nest[np.random.permutation(n)],nest[np.random.permutation(n)]),np.multiply(.01,np.random.random())).copy()

        # Yeni çözüm
        new_nest=np.add(nest,np.multiply(stepsize,K[0])).copy()

        # Cozum degerlerinin alt ve ust sinirlarinin uygulanmasi(Rastgele uretildiginden siniri asmis olabilir)
        for i in range(n):
            for j in range(dimension):                   
                if new_nest[i][j]<min_deger:
                    new_nest[i][j]=min_deger
                if new_nest[i][j]>max_deger:
                    new_nest[i][j]=max_deger
        return new_nest
    
    # Random walk yaparak gugukkuslarini bul
    def Guguk_Kusu_Bul(self,nest,best,min_deger,max_deger):
        n=len(nest)
        dimension=len(nest[0])

        # Levy exponent ve Levy coefficient
        beta=3/2
        sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        for i in range(len(nest)):
            s = nest[i]
            u=np.multiply(np.random.randn(dimension),sigma)
            v=np.random.randn(dimension)
            step=np.divide(u,np.power(np.abs(v),np.divide(1.,beta)))

            # Eger cozum en iyi cozumse degistirme
            stepsize=np.multiply(.01,np.multiply(step,np.subtract(best,nest[i]))).copy()

            # Random walk
            s=(np.add(s,np.multiply(stepsize,np.random.randn(dimension))))[:]

            # Cozum degerlerinin alt ve ust sinirlarinin uygulanmasi(Rastgele uretildiginden siniri asmis olabilir)
            for j in range(dimension):                   
                if s[j]<min_deger:
                    s[j]=min_deger
                elif s[j]>max_deger:
                    s[j]=max_deger
            nest[i]=s.copy()
        return nest
    
    # Su anki en iyi cozumu bul
    def En_Iyi_Cozum(self, nest,newnest,fitness):

        # Tum yeni cozumlerin fonksiyonda aldigi degerleri hesaplayip en iyilerle degistir
        for i in range(len(fitness)):
            fnew=self.Test_Fonksiyonu(newnest[i])
            if fnew<fitness[i]:
                fitness[i]=fnew
                nest[i]=newnest[i].copy()

        # En iyi cozumu ve indexini tut
        fmin=np.amin(fitness)
        index=np.argmin(fitness)
        best=nest[index]
        return (fmin,best,nest,fitness)
    
    
    def Test_Fonksiyonu(self,x): # Algoritmamıza verecegimiz test fonksiyonlarını belirlediğimiz kısım.
        
        if(self.sec == 1): # Beale Fonksiyonu
            return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
        if(self.sec == 2): # Ackley Fonksiyonu
            return -math.exp(-math.sqrt(0.5 * sum([i ** 2 for i in x]))) - math.exp(0.5 * sum([math.cos(i) for i in x])) + 1 + math.exp(1)
        if(self.sec == 3): # Goldstein–Price Fonksiyonu
            return ((1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * (x[0] ** 2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * (x[1] ** 2))) * (30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * (x[0] ** 2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1] ** 2))))
        if(self.sec == 4): # Levi Fonksiyonu 
            return (math.sin(3*x[0]*math.pi))**2 + ((x[0] - 1)**2)*(1 + math.sin(3*x[1]*math.pi)**2) + ((x[1] - 1)**2)*(1 + math.sin(2*x[1]*math.pi)**2)
        
a=Guguk_Kusu_Algoritmasi()
