from pylogit import mixed_logit
from mnl import load_data
X, Y = load_data('data/data.npy')
mixed = mixed_logit.