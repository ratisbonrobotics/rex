from functools import partial
from typing import Optional
import os
import pickle

import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
from scipy.spatial.transform import Rotation as R

from tqdm.auto import tqdm

import pytinyrenderer

from renderer import CameraParameters as Camera
from renderer import LightParameters as Light
from renderer import ModelObject as Instance
from renderer import ShadowParameters as Shadow
from renderer import Renderer, merge_objects, transpose_for_display


