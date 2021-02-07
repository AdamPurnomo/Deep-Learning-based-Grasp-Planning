#%%
import numpy as np 
import matplotlib.pyplot as plt
from scipy import interpolate  

H = np.load(r'..\..\Output\History\.npy')
epoch = H[0,0:6] + 1
tcl = H[1, 0:6]
tml = H[2]
ta = H[3, 0:6]/100
vcl = H[4,0:6]
vml = H[5]
va = H[6, 0:6]/100

x = np.linspace(1,6,300)
tcl_spline = interpolate.make_interp_spline(epoch, tcl)
ta_splline = interpolate.make_interp_spline(epoch, ta)

vcl_spline = interpolate.make_interp_spline(epoch, vcl)
va_spline = interpolate.make_interp_spline(epoch, va)


plt.figure()
plt.plot(x, tcl_spline(x), label = 'train_loss')
plt.plot(x, ta_splline(x), label = 'train_accuracy')
plt.plot(x, vcl_spline(x), label = 'validation_loss')
plt.plot(x, va_spline(x), label = 'validation_accuracy')
plt.ylim(0,1.1)
plt.yticks(np.arange(0, 1, 0.05))
plt.title('Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()