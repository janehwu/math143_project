'''
Jane Wu
jhwu@hmc.edu

Math 143 Midterm Project Code
'''

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

from time import time

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets
from sklearn.cluster import KMeans

# Next line to silence pyflakes. This import is needed.
Axes3D

Fs = 2.**12
T = 1./Fs
L = 64
shift = 32

AUDIO_FILE = "data/test.wav"

'''
Load audio data into X matrix
'''
def loadData():
	Fs, y = scipy.io.wavfile.read(AUDIO_FILE)
	print "Sampled at", Fs
	T = 1./Fs
	x = y[:,0]
	t = np.arange(0,L,T,dtype=float)
	freq = np.arange(0,Fs/2-1,Fs/L) # Frequencies between 0-2048 Hz
	print "Length", len(freq)

	# Input data
	X = []
	num_frames = len(x)
	print "Num frames:", num_frames

	# Keeps track of highest frequency at beginning of each window
	start_times = [] # In seconds
	#highest_freqs = [] # In Hz

	# Shifting window
	print "num shifts", int(float(num_frames)/shift)
	n = int(float(num_frames)/shift)

	for i in range(n-1):
		start_times.append(t[0+shift*i])
		frame = x[(shift*i):(shift*i + L)] # Get L frames per window
		if len(frame) < 64:
			print (shift*i + L) - (shift*i)
			print "WRONG", len(frame), i
			break

		# Perform FFT
		xdft = np.fft.fft(frame)
		xdft = np.abs(xdft)
		xdft[0] = 0 # Throw away 0 Hz...
		X.append(xdft)

	return np.asarray(X)
		
'''
Test FFT with sin function
'''
def testFFT():
	t = np.arange(0,L,T,dtype=float)
	x = np.zeros(len(t))
	for i in range(len(t)):
		x[i] = np.sin(2*np.pi*440*t[i])

	xdft = np.fft.rfft(x)
	freq = np.arange(0,Fs/2-1,Fs/L)
	xdft = xdft[0:L/2]

	I = np.argmax(xdft)
	print "Max frequency at", freq[I]
	plt.scatter(freq,np.abs(xdft))
	plt.show()

'''
Write to WAV file
'''
def writeWAV(source_num, labels):
	Fs, y = scipy.io.wavfile.read(AUDIO_FILE)
	T = 1./Fs
	t = np.arange(0,L,T,dtype=float)
	n,d = y.shape
	print "frames", n

	source = [] # Single source only
	jump = 30

	for i in range(0,len(labels)-jump,jump): #For each window
		# Find mean label
		mean = 0
		label = -1
		for j in range(jump):
			mean += labels[i+j]
		mean = mean/float(jump)
		if mean < 0.5:
			label = 0
		else:
			label = 1
		#print label
		if label == source_num:
			for k in range(jump*shift):
				#print i*shift+k
				source.append(y[i*shift+k])

	source = np.asarray(source)
	print source.shape
		
	scipy.io.wavfile.write("output" + str(source_num) + ".wav", Fs, source)


def main():
	# Load data
	X = loadData()

	# Manifold learning (spectral embedding)
	n_points = len(X)
	n_neighbors = 5
	n_components = 2

	print "Spectral embedding"
	se = manifold.SpectralEmbedding()
	print "Transforming"
	t0 = time()
	Z = se.fit_transform(X)
	t1 = time()
	print "Time:", t1-t0

	# Reduce dimensionality
	Z = Z[:,:2]

	# K-means
	print "K-mean"
	clf = KMeans(n_clusters=2, random_state=42)
	y = clf.fit(Z)

	print "X:", X.shape
	print "y:", y.labels_.shape
	print "Z:", Z.shape

	# Get counts
	# counts = [0,0]
	# for y in y.labels_:
	# 	print y
	# 	counts[y] += 1

	# Write to files
	# writeWAV(0, y.labels_)
	# writeWAV(1, y.labels_)

	color = y.labels_
	plt.scatter(Z[:, 0], Z[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.title("SpectralEmbedding")

	plt.show()

main()