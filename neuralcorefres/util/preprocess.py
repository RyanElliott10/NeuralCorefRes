from typing import List
import numpy as np

Tensor = List[float]

def single_output(xdata: List[Tensor], ydata: List[Tensor]) -> List[Tensor]:
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

