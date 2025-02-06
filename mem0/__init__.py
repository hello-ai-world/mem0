import importlib.metadata

#__version__ = importlib.metadata.version("mem0ai")
__version__ = '1.0.49'
import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from mem0.client.main import MemoryClient, AsyncMemoryClient  # noqa
from mem0.memory.main import Memory  # noqa
