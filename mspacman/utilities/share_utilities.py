import numpy as np
import multiprocessing as mp
import ctypes
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)

def ndarray_base_to_np(base, shape, dtype=np.float32):
    shared_ndarray = np.ctypeslib.as_array(base.get_obj())
    return shared_ndarray.view(dtype).reshape(*shape)

def shared_ndarray_base(shape, dtype=np.float32):
    """
    Form a shared memory numpy array.
    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing
    """
    mysize = int(np.prod(shape))
    #TODO: Handle types other than 32-bit float or 64-bit complex
    if dtype is np.complex64:
        mysize *= 2
    shared_ndarray_base = mp.Array(ctypes.c_float, mysize)
    return shared_ndarray_base
