import numpy as np
import sep

from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import cv2
from utils import *
from filters import levels
from scipy.interpolate import make_interp_spline


rcParams['figure.figsize'] = [10., 8.]

#Open Fits
img = open_process_fits('test.fits')
stretch(img,0.30)
levels(img,3000,1,65536,1,1,1,1)
data = img.data
histogram, bins = np.histogram(data.astype('uint8').flatten(), bins=256, range=[0, 256])
plt.figure()
plt.bar(bins[:-1], histogram, width=1)
plt.xlabel('Niveaux de gris')
plt.ylabel('Fréquence')
plt.title('Histogramme de l\'image')
plt.show()

# If image has color
if len(data.shape) == 3:
    # Convert to gray using luminance
    data = data[0]*0.299 + data[1]*0.587 + data[2]*0.114
else:
    # Image is already gray
    data = data.astype(np.float64)

# Display image
fig, ax = plt.subplots()
plt.imshow(data, interpolation='nearest', cmap='gray', origin='lower')


# Calculate background
m, s = np.mean(data), np.std(data)
bkg = sep.Background(data)
bkg_image = bkg.back()
bkg_rms = bkg.rms()

# Substract background
data_sub = data - bkg

# Extract stars and order by flux
objects = sep.extract(data_sub, 1.5, err=bkg.globalrms)
objects.sort(order='flux')

# Restrict best 10% limited to 15 stars
result=[]
n = len(objects)
n10 = n-min(int(0.1 * n),15)

# plot an ellipse for each object
for i in range(n10,n):
    e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                width=6*objects['a'][i],
                height=6*objects['b'][i],
                angle=objects['theta'][i] * 180. / np.pi)
    e.set_facecolor('none')
    e.set_edgecolor('red')
    ax.add_artist(e)

    # Save pixel value around center
    res=[]
    for x in range(-10,10):
        res.append(data[int(objects['y'][i]),int(objects['x'][i]+x)])
    result.append(res)

# Display curves with smothed data
fig2, ax2 = plt.subplots()
x=np.linspace(-10,10,20)
x_smooth = np.linspace(x.min(), x.max(), 200)

for line in result:
    spl = make_interp_spline(x, line)
    y_smooth = spl(x_smooth)
    ax2.plot(x_smooth,y_smooth,'-')
plt.show()