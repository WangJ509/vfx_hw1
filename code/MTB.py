import cv2 as cv
import numpy as np

def buildMTBPyramid(images, numLevel, percentile, exclude_range):
    pyramids = []
    for image in images:
        pyramids.append([])
        for level in range(numLevel):
            median = np.percentile(image, percentile)
            ret, thresholdMap = cv.threshold(image, median, 255, cv.THRESH_BINARY)
            exclusionMap = cv.bitwise_not(cv.inRange(image, median-exclude_range, median+exclude_range))
            pyramids[-1].append((thresholdMap, exclusionMap))

            if level != numLevel-1:
                image = cv.resize(image, None, fx=0.5, fy=0.5)

    return pyramids

def translateImage(image, x_shift, y_shift):
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

def alignmentMTBPair(pyramid1, pyramid2, numLevel, curLevel = 0):
    '''
    This implementation is different from the paper. The paper implementation uses one bit to represent a pixel, but our implementation uses one byte.
    '''

    if curLevel < numLevel - 1:
        cur_shift = alignmentMTBPair(pyramid1, pyramid2, numLevel, curLevel+1)
        cur_shift = cur_shift[0] * 2, cur_shift[1] * 2
    else:
        cur_shift = 0, 0

    thresholdMap1, exclusionMap1 = pyramid1[curLevel]
    thresholdMap2, exclusionMap2 = pyramid2[curLevel]

    min_error = thresholdMap1.shape[0] * thresholdMap1.shape[1]

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            x_shift = cur_shift[0] + i
            y_shift = cur_shift[1] + j

            shiftedThresholdMap2 = translateImage(thresholdMap2, x_shift, y_shift)
            shiftedExclusionMap2 = translateImage(exclusionMap2, x_shift, y_shift)

            diffMap = cv.bitwise_xor(thresholdMap1, shiftedThresholdMap2)
            diffMap = cv.bitwise_and(diffMap, exclusionMap1)
            diffMap = cv.bitwise_and(diffMap, shiftedExclusionMap2)

            error = cv.countNonZero(diffMap)
            if error < min_error:
                shift_ret = x_shift, y_shift
                min_error = error

    assert 'shift_ret' in locals(), "Error in calculating best shift"

    return shift_ret


def alignmentMTB(images, reference_image_index=None, grayscale=False, shift_bits=6, percentile=50, exclude_range=4):
    '''
    Median Threshold Bitmap alignment algorithm.
    Arguments:
        images: a list of images
        reference_image_index: the index of reference image (which will not be shifted)
        grayscale: True if the inputs are grayscale images
        shift_bits: (shift_bits + 1) equals the number of pyramid levels
        percentile: the percentile for creating threshold bitmap
        exclude_range: the (additive) range for computing exclusion bitmap
    Return value:
        A numpy array of aligned images
    '''

    if reference_image_index is None:
        reference_image_index = len(images) // 2

    if not grayscale:
        # 0.299 * R + 0.587 * G + 0.114 * B
        # These ratios are different from those proposed by the paper
        grayImages = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]
    else:
        grayImages = images

    pyramids = buildMTBPyramid(grayImages, shift_bits + 1, percentile, exclude_range)

    alignedImages = []
    for i in range(len(images)):
        if i == reference_image_index:
            alignedImages.append(images[i])
        else:
            x_shift, y_shift = alignmentMTBPair(pyramids[reference_image_index], pyramids[i], shift_bits + 1)
            alignedImages.append(translateImage(images[i], x_shift, y_shift))

    return np.array(alignedImages)


if __name__ == '__main__':
    images = [cv.imread(str(i) + '.jpeg') for i in [1, 2, 3, 4]]

    reference_index = 2

    alignedImages = alignmentMTB(images, reference_image_index=reference_index)
    
    # compare with opencv's implementation
    alignMTB = cv.createAlignMTB()
    for i in range(len(images)):
        if i == reference_index:
            print("ref img")
        else:
            img1 = cv.cvtColor(images[reference_index], cv.COLOR_BGR2GRAY)
            img2 = cv.cvtColor(images[i], cv.COLOR_BGR2GRAY)
            print(alignMTB.calculateShift(img1, img2))

    pass
