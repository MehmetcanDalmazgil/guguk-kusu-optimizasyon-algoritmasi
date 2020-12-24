##################### BENZETİMLİ TAVLAMA #####################

################## TEST FONKSİYONLARI ##################
# Beale Fonksiyonu (fmin(3,0.5) = 0)
# Ackley Fonksiyonu (fmin(0,0) = 0)
# Goldstein–Price Fonksiyonu (fmin(0,-1) = 3)
# Levi Fonksiyonu (fmin(1,1) = 0)

import math,random
import matplotlib.pyplot as plt

best = 0 # Fonksiyon sonucu
t = 100000 # Başlangıç sıcaklığı
m = 40 # Döngü başına deneme sayısı
x1 = 1 # Çözüm parametreleri x1 = x
x2 = 1 # Çözüm parametreleri x2 = y
alpha = 0.99 # Daha kötü çözümü kabul etme olasılığı

# Seçilen test fonksiyonuna göre üretilecek random sayıların sınırlarının belirlendiği ve üretildiği fonksiyon.
def randomState(a, b, sec): 
    if(sec == 1):
        min_deger = -4.5
        max_deger = 4.5
    elif(sec == 2):
        min_deger = -5
        max_deger = 5
    elif(sec == 3):
        min_deger = -4.5
        max_deger = 4.5
    elif(sec == 4):
        min_deger = -10
        max_deger = 10

    x1 = a * random.uniform(min_deger,max_deger)
    x2 = b * random.uniform(min_deger,max_deger)
    if(min_deger <= x1 <=max_deger and min_deger <= x2 <=max_deger):
        return x1, x2
    else:
        return randomState(a, b,sec)

# Test fonksiyonunun secilecegi kısım
print("\n")
print("1 --> Beale Fonksiyonu (fmin(3,0.5) = 0)")
print("2 --> Ackley Fonksiyonu (fmin(0,0) = 0)")
print("3 --> Goldstein–Price Fonksiyonu (fmin(0,-1) = 3)")
print("4 --> Levi Fonksiyonu (fmin(1,1) = 0)")
print("\n")
kontrol = True
sec = 0
while(kontrol):
    sec = int(input("Kullanmak istediginiz test fonksiyonunun numarasını giriniz : "))
    if(sec != 1 and sec != 2 and sec != 3 and sec != 4):
        print("Gecerli bir değer giriniz.")
    else:
        kontrol = False
print("\n")
adim = 1
best_tablo = []
best_x = []
best_y = []

# Sıcaklık 0 olana kadar yani soğuma işlemi gerçekleşene kadar çalışacak olan döngü.
while (t > 0.01):
    i = 0
    # Döngü başına düşen deneme sayısı(m) kadar kullanılan test fonksiyonu için bulunduğu sıcaklık değerindeki en iyi sonucu veren çözüm değerlerini bulan döngü 
    while (i <= m):
        a, b = randomState(x1, x2,sec) # Bulunulan sıcaklıkta en iyi çözüm değerlerinin bulunması için deneme sayısı kadar kullanılan test fonksiyonunun değer sınırlarına göre rastgele çözüm değerleri üretilen kısım. 
        
        # Kullanılacak test fonksiyonunun formulünün algoritmaya verildiği kısım.
        if(sec == 1): # Beale Fonksiyonu
            t_fonk = (1.5 - a + a*b)**2 + (2.25 - a + a*b**2)**2 + (2.625 - a + a*b**3)**2
        elif(sec == 2): # Ackley Fonksiyonu
            x = []
            x.append(a)
            x.append(b)
            t_fonk = -math.exp(-math.sqrt(0.5 * sum([i ** 2 for i in x]))) - math.exp(0.5 * sum([math.cos(i) for i in x])) + 1 + math.exp(1)
        elif(sec == 3): # Goldstein–Price Fonksiyonu
            t_fonk = ((1 + (a + b + 1)**2 * (19 - 14*a + 3*(a**2) - 14*b + 6*a*b + 3*(b**2))) * (30 + (2*a - 3*b)**2 * (18 - 32*a + 12*(a**2) + 48*b - 36*a*b + 27*(b**2))))
        elif(sec == 4): # Levi Fonksiyonu
            t_fonk = (math.sin(3*a*math.pi))**2 + ((a - 1)**2)*(1 + math.sin(3*b*math.pi)**2) + ((b - 1)**2)*(1 + math.sin(2*b*math.pi)**2)
       
        #  Önceki iterasyondaki fonksiyon sonucuyla şu an elde edilmiş fonksiyon sonucunu karşılaştıran ve çözüm değerleriyle sonuç değerini buna göre güncelleyen koşul ifadeleri.
        if best > t_fonk:
            x1 = a
            x2 = b
            best = t_fonk        
        elif(math.exp( (best - t_fonk) / t ) > random.random()):
            x1 = a
            x2 = b
            best = t_fonk

        # Sıcaklığın alpha sabitine bağlı olarak düşürüldüğü kısım.
        t = alpha*t
        i = i + 1
        
    print(f"Cycle: {adim} with Temperature: f({x1},{x2}) = {best}" )
    print("------------------------------------------------------")
    
    # Her iterasyon sonucu çözüm değerlerini ve fonksiyon sonucunu algoritma sonunda görselleştirmek için listeye eklendiği kısım.
    best_x.append(x1)
    best_y.append(x2)
    best_tablo.append(best)

    adim = adim + 1

# Benzetimli Tavlama sonucu test fonksiyonu için bulunan en iyi çözüm değerleri ile fonksiyon sonucunun ekrana yazdırıldığı kısım.
print(f"En iyi Çözüm (x) = {x1}")
print(f"En iyi Çözüm (y) = {x2}")
print(f"Fonksiyon Degeri f({x1},{x2}) = {best}")

# İterasyonlar sonucu oluşsan çözüm değerleri ve fonksiyon sonuçlarının görselleştirildiği kısım.
plt.grid()
plt.plot(best_x,best_y, 'D', color="b")

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(best_tablo,'r.-')
ax1.legend(['f_min'])
ax2 = fig.add_subplot(212)
ax2.plot(best_x,'b.-')
ax2.plot(best_y,'g--')
ax2.legend(['x','y'])
plt.show()

#################################################################################################
