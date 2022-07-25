from qboot import *

def c(i,s):
    return (maj(4*i+2*s) + 1j * maj(4*i+2*s+1))/2
def cd(i,s):
    return (maj(4*i+2*s) - 1j * maj(4*i+2*s+1))/2
def n(i,s):
    return cd(i,s) @ c(i,s)

L = 3
t, U = 1., 1.
H = - t * sum(sum(cd(i,s) @ c((i+1)%L,s)+cd((i+1)%L,s) @ c(i,s) for s in [0,1]) for i in range(L)) \
    + U * sum(n(i,0) @ n(i,1) for i in range(L))
rho = qboot(H, n=11, show_progress=True)
print(rho.inner(H).real)