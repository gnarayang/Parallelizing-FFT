import sys
import numpy
from scipy.io import wavfile

argument = sys.argv
print sys.argv[1]
fs, data = wavfile.read(str(sys.argv[1]))

file1 = open("../utils/input.dat", "w+")
file1.write(str(fs) + '\n')
file1.write(str(1 << int(numpy.log2(data.size))) + '\n')

for x in range(0, 1 << int(numpy.log2(data.size))):
    file1.write(str(data[x]) + '\n')