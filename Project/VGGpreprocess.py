import decimal
import numpy
import math
import logging


def round_half_up(num):
   return int(numpy.floor(num + 0.5))


def rolling_window(arr, window):  # Reshape a  numpy array into a rolling window of a given size and dim,
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window) # Replaces last dimension with two dimensions, one of which is the window
    strides = arr.strides + (arr.strides[-1],) # Keeps the same strides, but adds a new one for the window dimension
    return numpy.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides) # Take every step'th window


def framesig(sig, frameLen, frameStep, winfunc=lambda a: numpy.ones((a,))):
    #Frame a signal into overlapping frames.

    #param sig: Input audio signal
    #param frameLen: length of each frame measured in samples.
    #param frameStep: Gap between two frames in samples from BEGINNING TO BEGINNING.
    #param winfunc: the filter window.
    #param strideTrick: use stride trick to compute the rolling window and window multiplication faster
    #returns: an array of frames. Size is NUMFRAMES by frameLen.

    signalLen = len(sig)
    frameLen = int(round_half_up(frameLen))
    frameStep = int(round_half_up(frameStep))
    if signalLen <= frameLen:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * signalLen - frameLen) / frameStep)) # LV

    padlen = int((numframes - 1) * frameStep + frameLen)

    padsignal = numpy.concatenate((sig, numpy.zeros((padlen - signalLen,)))) # filling with zeros if not big enough 
    
    #The following can be done with stride tricks, but results are different, and the difference is not fully understood
    #win = winfunc(frameLen) # Rectangular window by default on the frame
    ####frames = rolling_window(padsignal, window=frameLen)


    indices = numpy.tile(numpy.arange(0, frameLen), (numframes, 1)) + numpy.tile( numpy.arange(0, numframes * frameStep, frameStep), (frameLen, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    frames = padsignal[indices]
    win = numpy.tile(winfunc(frameLen), (numframes, 1))

    return frames * win


def deframesig(frames, siglen, frameLen, frameStep, winfunc=lambda a: numpy.ones((a,))):
    #Does overlap-add procedure to undo the action of framesig.

    #param frames: the array of frames.
    #param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    #param frameLen: length of each frame in samples.
    #param frameStep: Gap between two frames in samples from BEGINNING TO BEGINNING.
    #param winfunc: the filter window
    #returns: a 1-D signal.

    frameLen = round_half_up(frameLen)
    frameStep = round_half_up(frameStep)
    numframes = numpy.shape(frames)[0] # the rows indices represent the number of frames
    assert numpy.shape(frames)[1] == frameLen, 'The length of the frames does not match frameLen'

    #This manually creates the indices matrix and pads it
    # Each row will be a frame of the same step (by transposing the arragement of all frames by steps), to each column we add the frame indices itself, which will result to the actual indices
    indices = numpy.tile(numpy.arange(0, frameLen), (numframes, 1)) + numpy.tile(
        numpy.arange(0, numframes * frameStep, frameStep), (frameLen, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    padlen = (numframes - 1) * frameStep + frameLen

    if siglen <= 0: siglen = padlen

    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    window = winfunc(frameLen)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[indices[i, :]] + window + 1e-15  # adds a little bit to not divide by zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def preemphasis(signal, coeff=0.95):
    return numpy.append(signal[0], signal[1:] - coeff*signal[:-1]) # High-pass filter y[n]=x[n]−αx[n−1]
