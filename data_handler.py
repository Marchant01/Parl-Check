import numpy as np
import pandas as pd

from main import get_all_members

members = pd.json_normalize(get_all_members()['personlista']['person'])

members.info()
