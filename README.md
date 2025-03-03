# ssgw

A Python implementation of SSGW (Steady Surface Gravity Waves)
by [Clamond and Dutykh (2018)](https://doi.org/10.1017/jfm.2018.208).

The original (MATLAB) implementation is available
[here](https://www.mathworks.com/matlabcentral/fileexchange/61499-surface-gravity-waves).

This Python implementation is a close translation of the original MATLAB code,
and thus the original license is included in [LICENSE.matlab](LICENSE.matlab).

## Installation

```
pip install git+https://github.com/wavesgroup/ssgw
```

## Usage

```python
import numpy as np
from ssgw import SSGW

wave = SSGW(kd=np.inf, kH2=0.1)
x, z = wave.zs.real, wave.zs.imag
u, w = wave.ws.real, wave.ws.imag
```

## Notes

* This program computes waves of arbitrary length for all heights up to about
99% of the maximum one. It is not designed to compute (near) limiting waves.

* The output quantities are dimensionless with the following scaling:
  * In deep water: `rho = g = k = 1`.
  * In finite depth: `rho = g = d = 1`.

## Examples

1. To compute a wave of steepness `kH2 = 0.3` in infinite depth:
```python
wave = SSGW(np.inf, 0.3)
```

2. To compute a cnoidal wave with height-over-depth=0.5 and length-over-depth=100:
```python
Hd = 0.5
Ld = 100
kd = 2 * np.pi / Ld
kH2 = np.pi * Hd / Ld
wave = SSGW(kd, kH2)
```

3. For steep and long waves, the default number of Fourier modes must be
increased. For instance, in order to compute a cnoidal wave with height-over-depth=0.7 and
length-over-depth=10000:
```python
Hd = 0.7
Ld = 10000
kd = 2 * np.pi / Ld
kH2 = np.pi * Hd / Ld
wave = SSGW(kd, kH2, 2**19)
```

## Issues?

Please report any issues [here](https://github.com/wavesgroup/ssgw/issues).