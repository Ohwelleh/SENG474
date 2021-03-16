import sys
from math import sqrt

error = float(sys.argv[1])
sampleSize = float(sys.argv[2])

interval = 1.96 * sqrt((error * (1 - error)) / sampleSize)

print(f'Confidence Interval: {interval}')