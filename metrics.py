# Copyright 2014, 2015, 2016, 2017 Matt Shannon

# This file is part of mcd.
# See `License` for details of license and warranty.

import math
import numpy as np

# 范数为1距离
def sqCepDist(x, y):
    diff = x - y
    return np.inner(diff, diff)

# 范数为2的距离
def eucCepDist(x, y):
    diff = x - y
    return math.sqrt(np.inner(diff, diff))

# log处为自然对数
# 加了修正长处后的开根号的范数为1的距离？
logSpecDbConst = 10.0 / math.log(10.0) * math.sqrt(2.0)
def logSpecDbDist(x, y):
    diff = x - y
    return logSpecDbConst * math.sqrt(np.inner(diff, diff))
