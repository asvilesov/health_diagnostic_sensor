from illum_models import *
import matplotlib.pyplot as plt
k = 30

waves = [420 + 10*l for l in range(33)]


print(skin_reflect_arr.shape)
plt.figure(1)
for i in range(0, 256, k):
    plt.plot(waves, skin_reflect_arr[:,i,1])
plt.legend([str(round(j*5/256+2,2)) + "%" for j in range(0,256,k)])
plt.title('Fixed Melanin at 1.3%, varying Hemoglobin')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Reflectance Spectra')

plt.figure(2)
for i in range(0, 256, k):
    plt.plot(waves, skin_reflect_arr[:,1,i])
plt.legend([str(round(j*43/256+1.3,2)) + "%" for j in range(0,256,k)])
plt.title('Fixed Hemoglobin at 2%, varying Melanin')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Reflectance Spectra')

plt.figure(3)
plt.plot(waves, illumA)
plt.title('Illuminant A - Tungsten, Spectral Power Distribution')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative Power')

plt.figure(4)
plt.plot(waves, spec_senstivity[3,0:33], c='r')
plt.plot(waves, spec_senstivity[3,33:66], c='g')
plt.plot(waves, spec_senstivity[3,66:99], c='b')
plt.legend(["Red Channel", "Green Channel", "Blue Channel"])
plt.title('Camera Spectral Sensitivties')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Absorbtion Spectra')

plt.figure(5)
plt.plot(waves, skin_reflect_arr[:,125,125])
plt.title('Typical Reflectance Spectra of Skin')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Reflectance Spectra')

plt.show()