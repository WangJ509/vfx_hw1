import cv2
import numpy as np
import random
import argparse
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
from MTB import alignmentMTB


class HDR():
    def __init__(self, sourceDirectory=".", l=600, n_samples=500, random_seed=0):
        self.sourceDirectory = sourceDirectory
        self.l = l
        self.n_samples = n_samples
        self.random_seed = random_seed

    def weight(self, z):
        if z <= 127:
            return z + 1
        return np.float32(256 - z)

    def weights(self, Z):
        weights = np.concatenate(
            (np.arange(1, 129), np.arange(1, 129)[::-1]), axis=0)
        return weights[Z].astype(np.float32)

    def sampleZ(self, Z):
        random.seed(self.random_seed)
        N, P = Z.shape
        indices = random.sample(range(N), self.n_samples)
        return Z[indices, :]

    def constructZ(self, images, color):
        images = images[..., color]
        Z = images.reshape(len(images), -1)
        return np.swapaxes(Z, 0, 1)

    def plotResponseCurves(self, gs):
        channels = len(gs)
        fig, ax = plt.subplots(1, channels, figsize=(5 * channels, 5))

        colors = ['blue', 'green', 'red']

        for c in range(channels):
            ax[c].plot(gs[c], np.arange(256), c=colors[c])
            ax[c].set_title(colors[c])
            ax[c].set_xlabel('E: Log Exposure')
            ax[c].set_ylabel('Z: Pixel Value')
            ax[c].grid(linestyle=':', linewidth=1)

        fig.savefig(os.path.join(self.sourceDirectory, "response_curve.jpeg"))

    def process(self, images, exposureTimes):
        B = np.log(exposureTimes)

        height, width = images.shape[1:3]
        print(f"Image resolution: {height}x{width}")

        result = np.zeros((height, width, 3), np.float32)

        gs = []
        for color in range(3):
            Z = self.constructZ(images, color)
            ZSample = self.sampleZ(Z)
            g = self.processOneColor(ZSample, B)
            gs.append(g)
            radiance = self.reconstructRadiance(Z, B, g)
            result[..., color] = radiance.reshape(height, width)

        self.plotResponseCurves(gs)
        return result

    def processOneColor(self, Z, B):
        N, P = Z.shape
        n = 256
        A = np.zeros((N*P + n+1, n+N), dtype=np.float32)
        b = np.zeros(len(A), np.float32)

        k = 0
        for i in range(N):
            for j in range(P):
                w = self.weight(Z[i][j])
                A[k][Z[i][j]] = w
                A[k][n+i] = -w
                b[k] = w * B[j]
                k += 1

        A[k, 128] = 1
        k += 1

        for i in range(n-2):
            val = self.l * self.weight(i + 1)
            A[k][i] = val
            A[k][i+1] = -2 * val
            A[k][i+2] = val
            k += 1

        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        g = x[:n]
        return g

    def reconstructRadiance(self, Z, B, g):
        N, P = Z.shape

        numerator = np.zeros(N)
        denominator = np.full(N, 1e-8)
        for j in range(P):
            lnT = np.full(N, B[j])
            weights = self.weights(Z[:, j])
            denominator += weights
            numerator += weights*(g[Z[:, j]] - lnT)
        E = np.exp(numerator / denominator)
        return E


def alignImages(images):
    return alignmentMTB(images)


def readFileAndExposureTimes(imageFileNames):
    exposureTimes = []
    # extract exposure time from exif information
    for fileName in imageFileNames:
        img = Image.open(fileName)
        exif = {ExifTags.TAGS[k]: v for k,
                v in img._getexif().items() if k in ExifTags.TAGS}
        exposureTimes.append(exif['ExposureTime'])
    exposureTimes = np.array(exposureTimes, dtype=np.float32)
    print(f"Exposure times: {exposureTimes}")

    images = np.array([cv2.imread(fileName) for fileName in imageFileNames])
    return images, exposureTimes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--source", help="image source directory path, default: data", default="data")
    parser.add_argument(
        "-r", "--random", help="random seed", default=0, type=int)
    parser.add_argument(
        "-n", "--number_sample", help="number of samples", default=100, type=int)
    args = parser.parse_args()
    sourceDirectory = args.source
    randomSeed = args.random
    number_sample = args.number_sample

    jpegFiles = glob.glob(sourceDirectory + "/*.jpeg")
    imageFileNames = []
    for file in jpegFiles:
        if "result" in file:
            continue
        if "response_curve" in file:
            continue
        imageFileNames.append(file)

    print(imageFileNames)
    images, exposureTimes = readFileAndExposureTimes(imageFileNames)

    images = alignImages(images)

    hdr = HDR(sourceDirectory, n_samples=number_sample, random_seed=randomSeed)
    result = hdr.process(images, exposureTimes)
    cv2.imwrite(os.path.join(sourceDirectory, 'result.hdr'), result)

    tonemap1 = cv2.createTonemap(gamma=2.2)

    merge_debevec = cv2.createMergeDebevec()
    hdr_debevec = merge_debevec.process(images, exposureTimes)
    # cv2.imwrite('debevec.hdr', hdr_debevec)
    res_debevec = tonemap1.process(hdr_debevec)
    cv2.imshow("opencv", res_debevec)

    res = 1.8 * tonemap1.process(result.copy())
    cv2.imwrite('result.png', 255 * res)

    cv2.imshow("JingMei", res)
    # cv2.setWindowProperty("JingMei", cv2.WND_PROP_TOPMOST, 1)

    cv2.waitKey(0)
