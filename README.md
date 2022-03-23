# VFX Project1 Report

|            | Name   | Student ID |
| ---------- | ------ | ---------- |
| Teammate 1 | 王靖傑 | r10944074  |
| Teammate 2 | 林首志 | r10922088  |

## Results
### Result HDR image (tone-mapped)
![](https://i.imgur.com/Hf99nrq.jpg)

### Recovered camera response curve
![](https://i.imgur.com/2SqVCcU.jpg)


## How to run the program

### Install dependencies

To install dependencies, run the following command:
```
pip install -r requirements.txt
```

### Command to run the program
```
python3 hdr.py [-h] [-s SOURCE] [-r RANDOM] [-n NUMBER_SAMPLE]

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE, --source SOURCE
                        image source directory path, default: data
  -r RANDOM, --random RANDOM
                        random seed
  -n NUMBER_SAMPLE, --number_sample NUMBER_SAMPLE
                        number of samples
```

- `-s` specifies the directory of input images
- `-r` specifies the random seed to get sample pixels
- `-n` specifies the number of sample pixels

For example, to get the result of our test images, run
```
python3 hdr.py -s data -n 500
```

## Contributions

This is what we have done in this project:
1. HDR (High Dynamic Range) algorithm based on Debevec's method
2. MTB alignment algorithm
3. Extract **EXIF** information from input images to get the exposure times.
4. Fine-tuned parameters like `lambda` and `number_sample` in Debevec's method, and the parameters of tone-mapping.

The test images were taken by **iphone 13 pro** with different exposure times (shutter speed).
The source images are stored in `data` (`1.jpeg` ~ `4.jpeg`)
The result image is stored in `data/result.jpeg`

### Debevec's method
Debevec's method was propsed in [this paper](http://www.csie.ntu.edu.tw/~cyy/courses/vfx/papers/Debevec1997RHD.pdf). The key idea is to get the original radiance given several source images with different exposure times. 
The following is the steps that we implement this algorithm
- Read input images by `imread()`, this function returns a ndarray with shape `(height, width, color)`
- Flatten those images into 1 dimension. Construct `Z`, where `Z[i][j]` means the pixel value of $i^{th}$ pixel and $j^{th}$ image.
- Sample `N` pixels from `Z`, where `N` can be execution argument and the default value is $500$.
- Construct `A` and `b` matrix and use `np.linalg.lstsq` to get the least square solution. The detail construction of `A` and `b` refers to the paper.
- After solving the least square solution, we get `g` from the first 256 elements.
- We can now calculate the radiance based on `g` and weighted by the source pixel values.
- Do the above steps to three color channels (e.g. RGB)
- To get the camera response curve, we plot the value of `g` to x-axis and `arange(256)` to y-axis.

### MTB alignment

We implemented the algorithm described in the [original paper](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/papers/Ward2003FRI.pdf). The following is an overview of our implementation:

1. Select a reference image from the input images.
2. Convert the input images to grayscale images.
3. For each image, build an MTB pyramid.
    3-1. Find the median intensity value. (Our implementation also supports arbitrary percentile.)
    3-2. Use the median to create the binary image. (Thresholding)
    3-3. Create the exclusion bitmap according to the median and the specified exclusion range.
    3-4. Create a half-size image, which will be used to create the next level of the pyramid.
    3-5. Repeat *3-1* to *3-4* until the number of levels reaches the specified value.
4. Align the reference image to every other image and obtain the shifts.
    4-1. Start from the highest level of the pyramids (the level with smallest images).
    4-2. Iterate all 9 possibilities of shifting the non-reference image within 1-pixel distances and find the best shift. The best shift is the shift that makes the images align the best.
    4-3. Go to the lower level, multiply the shift values by 2, and do *4-2* again. Stop when we reach the lowest level.
5. Translate the non-reference images according to the calculated shifts to make the images aligned.

There are some differences between our implementation and the implementation of the paper.
- We used OpenCV's function for converting color images to grayscale images. OpenCV's formula is different:
    - The formula of the paper: $\text{grey} = (54\times\text{red}+183\times\text{green}+19\times\text{blue})\; /\; 256$
    - [The formula of OpenCV](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html): $\text{grey}=0.299\times\text{red}+0.587\times\text{green}+0.114\times\text{blue}$
- The implementation of the paper uses **one bit** to represent a pixel of a binary image. Our implementation uses **one byte**. Using one byte makes the implementation easier and the calculation time is acceptable on a modern budget laptop.
- The implementation of the paper builds the pyramids **when** aligning a pair of images. Our implementation builds the pyramids **before** aligning images, so the pyramid of the refenrece image will not be unnecessarily recalculated again and again.

We have tested the correctness of our implementation by comparing the calculated shifts of our implementation with those of [OpenCV's implementation](https://docs.opencv.org/4.x/d7/db6/classcv_1_1AlignMTB.html). We found no difference between these two.

## Difficulties
- Because we programmed in Python, it is not easy to implement MTB algorithm with packed bits (one bit for one pixel). Numpy and OpenCV don't have specialized functions to handle operations on packed bits. If we program these functions ourselves, the calculation will not be fast because Python is slow in itself.
- When calculating the radiance, the program runs very slow. We found the problem is that we use a `for` loop to process each pixel, which takes very long because the resolution is $3024$x$4032$. We improve this issue by using numpy ndarray to process **all** pixels at a time (numpy would do parallel computing for element-wise operations).
- At the beginning, the result image looks very "blurry"! The reason is that we first set `lambda=10`, which is too low in practice. We found this looking at the camera response curve and found that it is very non-smooth. We solve this problem by setting larger `lambda`.
