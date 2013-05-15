# Canonical example to test PILS read / write similarity

import Image
import numpy as np
import numpy.testing as npt

fmt = 'JPEG'

raw_im = Image.open('raw.jpg')

raw_im.save('im_save.jpg', format=fmt)

save_im = Image.open('im_save.jpg')

save_im.save('im_save2.jpg', format=fmt)

save_im2 = Image.open('im_save2.jpg')

save_im2.save('im_save3.jpg', format='BMP')
save_im3 = Image.open('im_save3.jpg')

# arrays
raw_im_array = np.array(raw_im)
save_im_array = np.array(save_im)
save_im2_array = np.array(save_im2)
save_im3_array = np.array(save_im3)

# compare
raw_save = (raw_im_array == save_im_array).all()
save_save2 = (save_im_array == save_im2_array).all()
save2_save3 = (save_im2_array == save_im3_array).all()

print "raw == save ", raw_save
print "save == save2 ", save_save2
print "save2 == save3 ", save2_save3
