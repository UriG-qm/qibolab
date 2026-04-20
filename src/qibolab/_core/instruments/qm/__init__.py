from qibolab._core.parameters import ConfigKinds

from . import components, controller
from .components import *
from .components.configs import QmConfigs
from .controller import *

ConfigKinds.extend([QmConfigs])

__all__ = []
__all__ += components.__all__
__all__ += controller.__all__
