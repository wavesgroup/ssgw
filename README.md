# ssgw

A Python implementation of SSGW (Steady Surface Gravity Waves)
by [Clamond and Dutykh (2018)](https://doi.org/10.1017/jfm.2018.208).

The original (MATLAB) implementation is available
[here](https://www.mathworks.com/matlabcentral/fileexchange/61499-surface-gravity-waves).

This Python implementation is a close translation of the original MATLAB code,
and thus the original license is included in [LICENSE.matlab](LICENSE.matlab).

## Installation

```
pip install ssgw
```

## Usage

```python
import numpy as np
import ssgw

wave = ssgw.SSGW(kd=np.inf, kH2=0.1)
```
