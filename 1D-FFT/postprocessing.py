import numpy
from scipy.io import wavfile

file1 = open("../utils/convert_to_wav.dat", "r+")

# Raw data has all the data from the file as a list
raw_data = file1.readlines()
# The first element of the file is the sampling rate
fs = int(raw_data[0][0:-1])
# The rest of the data is amplitudes
data = []
for i in range (1, len(raw_data)):
    data.append(int(raw_data[i][0:-1]))
data = numpy.array(data, numpy.int16)

wavfile.write('../utils/proccessed_output.wav', fs, data)
