
# coding: utf-8

# In[2]:

import numpy as np

get_ipython().magic(u'config InlineBackend.re = {}')
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from astropy.modeling.models import Gaussian2D
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.utils.data import download_file


# In[3]:

from astropy.io import fits
from astropy import wcs

hdu_list0 = fits.open('image000000.fits')
hdu_list0.info()


# In[4]:

image_data0 = hdu_list0[0].data


# In[5]:

print(type(image_data0))
print(image_data0.shape)


# In[6]:

hdu_list0.close()


# In[7]:

plt.imshow(image_data0, cmap='gray', vmin=13825, vmax=14025)
plt.colorbar()


# In[8]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout0 = Cutout2D(image_data0, position, size)
median=14638
plt.imshow(cutout0.data, origin='imagedata0', vmin=median-25, vmax=median+1000)


# In[9]:

print('Median:', np.median(cutout0.data))


# In[10]:

print(type(image_data0.flatten()))


# In[11]:

hdu_list1 = fits.open('image000001.fits')
hdu_list1.info()


# In[12]:

image_data1 = hdu_list1[0].data


# In[13]:

print(type(image_data1))
print(image_data1.shape)


# In[14]:

hdu_list1.close()


# In[15]:

plt.imshow(image_data1, cmap='gray', vmin=13850, vmax=13950)
plt.colorbar()


# In[16]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout1 = Cutout2D(image_data1, position, size)
median1=14519
plt.imshow(cutout1.data, origin='imagedata1', vmin=median1-25, vmax=median1+1000)


# In[17]:

print('Median:', np.median(cutout1.data))


# In[18]:

image_list = ['image000000.fits', 'image000001.fits']

image_concat = []
for image in image_list:
    image_concat.append(fits.getdata(image)/2)
    
final_image0 = np.zeros(shape=image_concat[0].shape)

for image in image_concat:
    final_image0 += image 


# In[19]:

plt.imshow(final_image0, cmap='gray')
plt.colorbar()


# In[20]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout_composite0 = Cutout2D(final_image0, position, size)
median=14527
plt.imshow(cutout_composite0.data, origin='final_image', vmin=median-25, vmax=median+1000)


# In[21]:

print('Median:', np.median(cutout_composite0.data))


# In[22]:

hdu_list2 = fits.open('image000002.fits')
hdu_list2.info()


# In[23]:

image_data2 = hdu_list2[0].data


# In[24]:

print(type(image_data2))
print(image_data2.shape)


# In[25]:

hdu_list2.close()


# In[26]:

plt.imshow(image_data2, cmap='gray', vmin=13850.0, vmax=13950.0)
plt.colorbar()


# In[27]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout2 = Cutout2D(image_data2, position, size)
median2=14037
plt.imshow(cutout2.data, origin='imagedata2', vmin=median2-25, vmax=median2+1000)


# In[28]:

print('Median:', np.median(cutout2.data))


# In[29]:

hdu_list3 = fits.open('image000003.fits')
hdu_list3.info()


# In[30]:

image_data3 = hdu_list3[0].data


# In[31]:

print(type(image_data3))
print(image_data3.shape)


# In[32]:

hdu_list3.close()


# In[33]:

plt.imshow(image_data3, cmap='gray', vmin=13850.0, vmax=13950.0)
plt.colorbar()


# In[34]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout3 = Cutout2D(image_data3, position, size)
median=14000
plt.imshow(cutout3.data, origin='imagedata3', vmin=median-25, vmax=median+1000)


# In[35]:

print('Median:', np.median(cutout3.data))


# In[36]:

final_spec = np.zeros(shape=image_list[0].shape)

for image in image_concat:
    final_spec += image


# In[38]:

hdulist6 = fits.open('image000006.fits')
hdulist6.info()


# In[39]:

image_data6 = hdulist6[0].data


# In[40]:

print(type(image_data6))
print(image_data6.shape)


# In[41]:

hdulist6.close()


# In[42]:

plt.imshow(image_data6, cmap='gray')
plt.colorbar()


# In[43]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout6 = Cutout2D(image_data6, position, size)
median=13992
plt.imshow(cutout6.data, origin='imagedata6', vmin=median-25, vmax=median+1000)


# In[44]:

print('Median:', np.median(cutout6.data))


# In[45]:

hdulist7 = fits.open('image000007.fits')
hdulist7.info()


# In[46]:

image_data7 = hdulist7[0].data


# In[47]:

print(type(image_data7))
print(image_data7.shape)


# In[48]:

hdulist7.close()


# In[49]:

plt.imshow(image_data7, cmap='gray')
plt.colorbar()


# In[50]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout = Cutout2D(image_data7, position, size)
median=14291
plt.imshow(cutout.data, origin='imagedata7', vmin=median-25, vmax=median+1000)


# In[51]:

hdulist8 = fits.open('image000008.fits')
hdulist8.info()


# In[52]:

image_data8 = hdulist8[0].data


# In[53]:

print(type(image_data8))
print(image_data8.shape)


# In[54]:

hdulist8.close()


# In[55]:

plt.imshow(image_data8, cmap='gray')
plt.colorbar()


# In[56]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout = Cutout2D(image_data8, position, size)
median=14291
plt.imshow(cutout.data, origin='imagedata8', vmin=median-25, vmax=median+1000)


# In[57]:

hdulist9 = fits.open('image000009.fits')
hdulist9.info()


# In[58]:

image_data9 = hdulist9[0].data


# In[59]:

print(type(image_data9))
print(image_data9.shape)


# In[60]:

hdulist9.close()


# In[61]:

plt.imshow(image_data9, cmap='gray')
plt.colorbar()


# In[62]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout = Cutout2D(image_data9, position, size)
median=13922
plt.imshow(cutout.data, origin='imagedata9', vmin=median-25, vmax=median+1000)


# In[63]:

print('Median:', np.median(cutout.data))


# In[64]:

hdulist10 = fits.open('image000010.fits')
hdulist10.info()


# In[65]:

image_data10 = hdulist10[0].data


# In[66]:

print(type(image_data10))
print(image_data10.shape)


# In[67]:

hdulist10.close()


# In[68]:

plt.imshow(image_data10, cmap='gray')
plt.colorbar()


# In[69]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout10 = Cutout2D(image_data10, position, size)
median=13824
plt.imshow(cutout10.data, origin='imagedata10', vmin=median-25, vmax=median+1000)


# In[70]:

print('Median:', np.median(cutout10.data))


# In[71]:

hdulist18 = fits.open('image000018.fits')
hdulist18.info()


# In[72]:

image_data18 = hdulist18[0].data


# In[73]:

print(type(image_data18))
print(image_data18.shape)


# In[74]:

print'Maximum:', np.max(image_data18[0:,300])

sorted_int = sorted(image_data18[0:,300])
largest_int = sorted_int[-1]
print(largest_int)
sec_larg_int = sorted_int[-2]
print(sec_larg_int)
thr_larg_int = sorted_int[-3]
print(thr_larg_int)


# In[75]:

hdulist18.close()


# In[76]:

plt.imshow(image_data18, cmap='gray', vmin=14040, vmax=14240)
plt.colorbar()


# In[77]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout18 = Cutout2D(image_data18, position, size)
median=14041
plt.imshow(cutout18.data, origin='imagedata18', vmin=median-1, vmax=median+200)


# In[78]:

print('Median:', np.median(cutout18.data))
print("Maximum:", np.max(cutout18.data))


# In[79]:

hdulist19 = fits.open('image000019.fits')
hdulist19.info()


# In[80]:

image_data19 = hdulist19[0].data


# In[81]:

print(type(image_data19))
print(image_data19.shape)


# In[82]:

hdulist19.close()


# In[83]:

plt.imshow(image_data19, cmap='gray')
plt.colorbar()


# In[84]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout19 = Cutout2D(image_data19, position, size)
median=14010
plt.imshow(cutout19.data, origin='imagedata19', vmin=median-1, vmax=median+200)


# In[85]:

print('Median:', np.median(cutout19.data))


# In[86]:

hdulist20 = fits.open('image000020.fits')
hdulist20.info()


# In[87]:

image_data20 = hdulist20[0].data


# In[88]:

print(type(image_data20))
print(image_data20.shape)


# In[89]:

hdulist20.close()


# In[90]:

plt.imshow(image_data20, cmap='gray')
plt.colorbar()


# In[91]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout20 = Cutout2D(image_data20, position, size)
median=13987
plt.imshow(cutout20.data, origin='imagedata20', vmin=median-1, vmax=median+200)


# In[92]:

print('Median:', np.median(cutout20.data))


# In[94]:

image_list = ['image000018.fits', 'image000019.fits', 'image000020.fits']

image_concat = []
for image in image_list:
    image_concat.append(fits.getdata(image)/3)
    
final_image1 = np.zeros(shape=image_concat[0].shape)

for image in image_concat:
    final_image1 += image 


# In[95]:

plt.imshow(final_image1, cmap='gray')
plt.colorbar()


# In[97]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,2069/2)
size = (2069,1000)
cutout_composite1 = Cutout2D(final_image1, position, size)
median=14011
plt.imshow(cutout_composite1.data, origin='final_image1', vmin=median-25, vmax=median+200)


# In[98]:

print('Median:', np.median(cutout_composite1.data))


# In[99]:

image_list1 = ['image000018.fits', 'image000019.fits', 'image000020.fits']

image_concat1 = []
for image in image_list1:
    image_concat1.append(fits.getdata(image)/3)
    
final_image1 = np.zeros(shape=image_concat[0].shape)

for image in image_concat1:
    final_image1 += image 


# In[100]:

plt.imshow(final_image1, cmap='gray')
plt.colorbar()


# In[101]:

y, x = np.mgrid[0:3216, 0:2069]
data = Gaussian2D(1, 50, 100, 10, 5, theta=0.5)(x, y)
position = (3216/2,50)
size = (2069,1000)
cutout_composite1 = Cutout2D(final_image1, position, size)
median=14011
plt.imshow(cutout_composite1.data, origin='final_image1', vmin=median-1, vmax=median+200)


# In[102]:

print'Maximum:', np.max(final_image1[0:,300])


# In[129]:

for m in range(0,550):
    print 'Maximum_1:', np.max(final_image1[0:350,m])
    print ' Maximum_c:', np.max(final_image1[351:680, m])
    print 'Maximum_r:', np.max(final_image1[681:,m])
    l = np.max(final_image1[0:350, m])
    c = np.max(final_image1[351:680, m])
    r = np.max(final_image1[681:, m])
    #print list(np.max(final_image1[0:350,m])


# In[161]:

for m in range(0,550):
    #list (np.max(final_image1[0:350,m]))
    print np.max(final_image1[0:350, m])
    #print '[%s]' % ', '.join( map(str, l))
    
    #c = np.max(final_image1[351:680, m])
    #r = np.max(final_image1[681:, m])


# In[154]:

import matplotlib.pyplot as plt
m = np.arange(0.,550., 1.)
plt.plot([m],[str (np.max(final_image1[0:350,m]))], 'r--', [m], [c], 'b--', [m], [r], 'ro')

