import numpy as np
import matplotlib.pyplot as plt

from pydwf import (DwfLibrary, DwfEnumConfigInfo, DwfAnalogOutIdle, DwfTriggerSource, 
                   DwfAnalogOutNode, DwfAnalogOutFunction, DwfAcquisitionMode,
                   DwfAnalogInTriggerType, DwfTriggerSlope, DwfState, DwfAnalogInFilter,
                   PyDwfError)
from pydwf.utilities import openDwfDevice