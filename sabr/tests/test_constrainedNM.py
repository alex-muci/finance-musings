""" test a couple of functions (global and non-)

    source: https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

from analytics.calibration.constrNM import constrNM
import numpy as np


def beale_fnct(x):
    """
    Beale function given by
    :math: f(x,y)=(1.5-x+xy)^2+(2.25-x+xy^2)^2+(2.625-x+xy^3)^2

    expected result:
    - global min: f(3,0.5)=0
    - search area: -4.5 <= x,y <= 4.5

    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    return (1.5-x[0]+x[0]*x[1])**2 \
        + (2.25-x[0]+x[0]*x[1]**2)**2 \
        + (2.625-x[0]+x[0]*x[1]**3)**2


def rosenbrock_fnct(x):
    """
    Rosenbrock function in 2-dim given by
    :math: f(x,y)=(1-x)^2+100(y-x^2)^2

    https://en.wikipedia.org/wiki/Test_functions_for_optimization .
    """
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2


# ... and test them
def test_beales():
    res = constrNM(beale_fnct, [0.5, 1], [-5, -5], [10, None], full_output=False)
#   print("result is {} and expected is {}" .format(res['xopt'], np.array([3., 0.5])))
    assert abs(res['xopt'] - np.array([3., 0.5])).sum() < 1E-3


def test_rosenbrock():
    res = constrNM(rosenbrock_fnct, [2.5, 2.5], [2, 2], [None, 3], full_output=True)
#   print("result is {} and expected is {}".format(res['xopt'], np.array([2., 3.])))
    assert abs(res['xopt'] - np.array([2., 3.])).sum() < 1E-3


# if __name__ == '__main__':
#     test_beales()
#     test_rosenbrock()
