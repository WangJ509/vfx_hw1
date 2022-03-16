import cv2 as cv
import numpy as np
from PIL import Image, ExifTags

def alignImages(images):
    alignMTB = cv.createAlignMTB()
    alignMTB.process(images, images)

    return images


imageFileNames = ['1.jpeg', '2.jpeg', '3.jpeg', '4.jpeg']
exposureTimes = []
# extract exposure time from exif information
for fileName in imageFileNames:
    img = Image.open(fileName)
    exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    exposureTimes.append(float(exif['ExposureTime']))
exposureTimes = np.array(exposureTimes, dtype=np.float32)
print (exposureTimes)

images = [cv.imread(fileName) for fileName in imageFileNames]
alignImages(images)
merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(images, times=exposureTimes.copy())

tonemap1 = cv.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())

cv.imshow("debevec", res_debevec)
# cv.setWindowProperty("test", cv.WND_PROP_TOPMOST, 1)

merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(images, times=exposureTimes.copy())

res_robertson = tonemap1.process(hdr_robertson.copy())
# cv.imshow("hdr_robertson", res_robertson)

cv.waitKey(0)