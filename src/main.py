import numpy as np
import matplotlib.pyplot as plt

from optic.comm.modulation import modulateGray, demodulateGray, grayMapping
#from optic.dsp.core import pnorm, upsample, firFilter, pulseShape, signal_power, phaseNoise
from optic.models.devices import mzm, iqm
from optic.utils import parameters, dBm2W
from optic.plot import eyediagram

print("hello world!")