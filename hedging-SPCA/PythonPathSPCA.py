"""
"Optimal Solutions for Sparse Principal Component Analysis"
by Alexandre d'Aspremont, Francis Bach, Laurent El Ghaoui
"""
from scipy import array, real, dot, column_stack, row_stack, append
import numpy

ra = numpy.random
la = numpy.linalg


# noinspection PyUnresolvedReferences,PyShadowingBuiltins,PyPep8Naming,SpellCheckingInspection
def path_spca(A, k):
    M, N = A.shape
    # Loop through variables
    As = ((A * A).sum(axis=0))
    # vmax = As.max()
    vp = As.argmax()
    subset = [vp]
    vars = []
    res = subset
    rhos = [(A[:, vp] * A[:, vp]).sum()]
    Stemp = array([rhos])
    for i in range(1, k):
        lev, v = la.eig(Stemp)
        vars.append(real(lev).max())
        vp = real(lev).argmax()
        x = dot(A[:, subset], v[:, vp])
        x /= la.norm(x)
        seto = list(range(0, N))
        for j in subset:
            seto.remove(j)
        vals = dot(x.T, A[:, seto])
        vals *= vals
        rhos.append(vals.max())
        vpo = seto[vals.argmax()]
        Stemp = column_stack((Stemp, dot(A[:, subset].T, A[:, vpo])))
        vbuf = append(dot(A[:, vpo].T, A[:, subset]), array([(A[:, vpo] * A[:, vpo]).sum()]))
        Stemp = row_stack((Stemp, vbuf))
        subset.append(vpo)
    lev, v = la.eig(Stemp)
    vars.append(real(lev).max())
    return vars, res, rhos


# **** Run quick demo ****
if __name__ == "__main__":

    # Simple data matrix with N=7 variables and M=3 samples
    k = 2  # target cardinality
    A = array([[1, 2, 3, 4, 3, 2, 1],
               [4, 2, 1, 4, 3, 2, 1],
               [5, 2, 3, 4, 3, 3, 1]]
              )

    # Call function
    # noinspection PyShadowingBuiltins
    vars, res, rhos = path_spca(A, k)
    print(res)
    print(vars)
    print(rhos)
