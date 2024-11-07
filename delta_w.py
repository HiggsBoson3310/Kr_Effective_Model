import math_util as MU
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
fsperau=2.4188843e-17/1.0e-15; auI = 3.50944e16; evperAU=27.2114079527e0
fsm1toeV = fsperau*evperAU

E1 = 26.84082546
E2 = 25.1694016
gam = 40/fsperau * 1/np.sqrt(4*np.log(2.0))

eew = np.linspace(0.5*(E1+E2)-np.sqrt(np.log(1e3)/(gam**2/(8*evperAU**2))),0.5*(E1+E2)+np.sqrt(np.log(1e3)/(gam**2/(8*evperAU**2))),300)


l1, = plt.plot(eew, np.real(MU.w(gam*((E2+E1-2*eew)/evperAU)/np.sqrt(8))))
l2, = plt.plot(eew, np.imag(MU.w(gam*((E2+E1-2*eew)/evperAU)/np.sqrt(8))))

ax_p = plt.axes([0.1,0.01,0.8,0.03],facecolor='blue')
slider = Slider(ax_p, "mult",26.5,27.5, valinit=E1)

def update(val):
    delta = slider.val
    plt.axvline(0.5*(delta+E2))
    l1.set_ydata(np.real(MU.w(gam*((E2+delta-2*eew)/evperAU)/np.sqrt(8))))
    l2.set_ydata(np.imag(MU.w(gam*((E2+delta-2*eew)/evperAU)/np.sqrt(8))))
    
    
slider.on_changed(update)
#plt.axvline(E2)
#plt.axvline(E1)

plt.show()