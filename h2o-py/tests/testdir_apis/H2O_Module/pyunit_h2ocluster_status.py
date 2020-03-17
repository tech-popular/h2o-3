from __future__ import print_function
import sys
sys.path.insert(1,"../../../")
from tests import pyunit_utils
import h2o

def h2ocluster_get_status():
    """
    Python API test: h2o.cluster().get_status()
    """
    table = h2o.cluster().get_status(detailed=True)

if __name__ == "__main__":
    pyunit_utils.standalone_test(h2ocluster_get_status)
else:
    h2ocluster_get_status()
