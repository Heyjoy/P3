import math


# models hyperParmeter
TrainTestSplitSize = 0.2
N_EPOCH = 20
Verbose = 1
BatchSize = 64
zeroSteeringCount = 3
#GaussianNoiseStddev = 1


# Imgae Process tuning paramter
IMGPath = '../data/IMG/'
CSVPath = '../data/driving_log.csv'
ImgShape = [160, 320, 3]
ResizedShape = [64, 64, 3]
cropBottom = math.floor(ImgShape[0]/6) #
cropTop = cropBottom * 2

AngleOffset = 0.25 # offset for left and right camera

## Image flip random
FilpProb = 0.5
## Brightness random
RandomBrightOffset = 0.25
## translate Image method parameter
x_trRange = int(ImgShape[1]/10) # 320 = 6.4*50
y_trRange = int(ImgShape[0]/10) # 160 = 6.4 *25

trShiftAngle = 0.4
