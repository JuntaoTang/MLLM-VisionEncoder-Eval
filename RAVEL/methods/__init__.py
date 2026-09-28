from methods.cca import CCA
from methods.gw import GW
from methods.mutualnn import MutualNN
from methods.ravel import RAVEL
from methods.rsa import RSA


METHODS = {
    method.name: method
    for method in (RSA, CCA, RAVEL, GW, MutualNN)
}


__all__ = ["METHODS", "RSA", "CCA", "RAVEL", "GW", "MutualNN"]
