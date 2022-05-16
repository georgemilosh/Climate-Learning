import matplotlib.pyplot as plt
import numpy as np
import stochrare as sr
import sys
sys.path.insert(1, '../CDS')
from ERA_Fields import*
np.random.seed(seed=100)

oup = sr.dynamics.diffusion1d.OrnsteinUhlenbeck1D(0, 1/5, 2)
reftraj = oup.trajectory(0., 0., T=1e8)
np.save('reftraj.npy',reftraj)
