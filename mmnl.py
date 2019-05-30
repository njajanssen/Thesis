import pylogit as pl
from mnl import load_data
import pandas as pd
from collections import OrderedDict
import numpy as np

X, Y = load_data('data/data.npy')
catsup = pd.read_csv('data/catsup_trainformat.csv')
# catsup = catsup.drop('Unnamed: 0', axis=1)
#get variable list
index_var_names = ['display','feature','price']
for col in index_var_names:
    catsup[col] = catsup[col].astype(float)
#specification
example_specification = OrderedDict()
example_names = OrderedDict()

# Note that the names used below are simply for consistency with
# the coefficient names given in the mlogit vignette.
for col in index_var_names:
    example_specification[col] = [[0, 1, 2, 3]]
    example_names[col] = [col]
# Provide the module with the needed input arguments to create
# an instance of the Mixed Logit model class.

# Note that "chid" is used as the obs_id_col because "chid" is
# the choice situation id.

# Currently, the obs_id_col argument name is unfortunate because
# in the most general of senses, it refers to the situation id.
# In panel data settings, the mixing_id_col argument is what one
# would generally think of as a "observation id".

# For mixed logit models, the "mixing_id_col" argument specifies
# the units of observation that the coefficients are randomly
# distributed over.
example_mixed = pl.create_choice_model(data=catsup,
                                       alt_id_col="alt",
                                       obs_id_col="chid",
                                       choice_col="chosen",
                                       specification=example_specification,
                                       model_type="Mixed Logit",
                                       names=example_names,
                                       mixing_id_col="id",
                                       mixing_vars=index_var_names)

# Note 2 * len(index_var_names) is used because we are estimating
# both the mean and standard deviation of each of the random coefficients
# for the listed index variables.
example_mixed.fit_mle(init_vals=np.zeros(2 * len(index_var_names)),
                      num_draws=600,
                      seed=123)

# Look at the estimated results
print(example_mixed.get_statsmodels_summary())


