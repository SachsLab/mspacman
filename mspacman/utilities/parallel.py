
from __future__ import division

import time
import types
import numpy as np
import multiprocessing as mp
import warnings
from . import share_utilities as sh

class Counter(object):
    def __init__(self, initval=0):
        self.val = mp.RawValue('i', initval)
        self.lock = mp.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value

class ParallelDummy(object):
    """
    This class is created to wrap a function such that it has the same iterface as the Parallel class.
    """
    function = {}

    def __init__(self, func, *args, **kwargs):

        warnings.warn("The multiprocessing is disabled! To enable multiprocessing, "+\
                      "specify 'ins_shape' and 'out_shape' for preallocating shared memory.")

        self.f_name = func.__name__
        self.function[self.f_name] = func
        self.kwargs = kwargs

        self.del_opt = None

    def __del__(self):
        self.kill(opt=self.del_opt)

    def kill(self, opt=None):
        if opt is not None:
            print("The object, {}, is deleted off the memory.".format(self))

        del self

    def result(self, *args, **kwargs):
        return self.function[self.f_name](*args, **self.kwargs)

class Parallel(object):
    """
    This class is created to wrap a function for multiprocessing.
    """
    function = {}
    def __init__(self, func, *args, nprocs=1, axis=0, flag=0, **kwargs):
        self.nprocs = nprocs
        self.del_opt = None

        self.f_name = func.__name__
        self.function[self.f_name] = func

        self.flag = flag
        self.axis = axis

        self.out_shape = kwargs.pop('out_shape')
        self.out_dtype = kwargs.pop('out_dtype')

        self.ins_shape = kwargs.pop('ins_shape')
        self.ins_dtype = kwargs.pop('ins_dtype')

        self.in1_shape = self.ins_shape[0]
        self.in2_shape = self.ins_shape[1]

        self.in1_dtype = self.ins_dtype[0]
        self.in2_dtype = self.ins_dtype[1]

        if self.nprocs > 1:

            # Create all shared memory arrays used
            in1_base = sh.shared_ndarray_base(self.in1_shape, dtype=self.in1_dtype)
            in2_base = sh.shared_ndarray_base(self.in2_shape, dtype=self.in2_dtype)
            out_base = sh.shared_ndarray_base(self.out_shape, dtype=self.out_dtype)

            self.in1 = sh.ndarray_base_to_np(in1_base, self.in1_shape, dtype=self.in1_dtype)
            self.in2 = sh.ndarray_base_to_np(in2_base, self.in2_shape, dtype=self.in2_dtype)
            self.out = sh.ndarray_base_to_np(out_base, self.out_shape, dtype=self.out_dtype)

            # Create slices to reconstruct output
            self.slices = []
            start_slice = 0
            for p in range(self.nprocs):
                slice_remain = self.out_shape[self.axis] - start_slice
                procs_remain = self.nprocs - p
                slice_i = int(np.ceil(slice_remain / procs_remain))
                self.slices += [slice(start_slice, start_slice+slice_i)]
                start_slice += slice_i

            # Create counters to synchronize processes
            self.in_counter = Counter() # for in1, since slice changed
            self.out_counter = [Counter() for n in range(self.nprocs)]

            self.procs = [mp.Process(target=self.process,
                                     args=(n, self.in_counter, self.out_counter[n], in1_base, in2_base, out_base),
                                     kwargs=kwargs)
                                     for n in range(self.nprocs)]

            for p in self.procs:
                p.daemon = True
                p.start()

    def __del__(self):
        self.kill(opt=self.del_opt)

    def kill(self, opt=None): # kill the multiprocess

        try:
            tmp = self.in_counter.val.value
            self.in_counter.val.value = -1
            [p.join() for p in self.procs]

            if opt is not None:
                print("All processes in {} are closed.".format(self))

        except (AttributeError, AssertionError):
            # Processes are not running
            pass

    def process(self, proc_i, in_counter, out_counter, in1_base, in2_base, out_base, **kwargs):

        in1 = sh.ndarray_base_to_np(in1_base, self.in1_shape, dtype=self.in1_dtype)
        in2 = sh.ndarray_base_to_np(in2_base, self.in2_shape, dtype=self.in2_dtype)
        out = sh.ndarray_base_to_np(out_base, self.out_shape, dtype=self.out_dtype)

        idx = [slice(None)] * len(self.out_shape)
        idx[self.axis] = self.slices[proc_i]

        tmp1 = in1[idx[:-1]]
        if self.flag == 0:
            tmp2 = in2
        elif self.flag == 1:
            tmp2 = in2[idx[:-1]]

        while in_counter.value() >= 0:
            while in_counter.value() >= 0 and out_counter.value() >= in_counter.value():
                time.sleep(0.001)

            if in_counter.value() >= 0:
                out[idx] = self.function[self.f_name](tmp1, tmp2, **kwargs)
                out_counter.increment()

    def result(self, *args, **kwargs):

        if self.nprocs > 1:

            self.in1[:,:] = args[0]
            self.in2[:,:] = args[1]

            self.in_counter.increment()

            while np.any(np.asarray([x.value() for x in self.out_counter]) < self.in_counter.value()):
                time.sleep(0.01)
            return self.out

        else:
            return self.function[self.f_name](*args, axis=-1, **kwargs)

    @staticmethod
    def check_nprocs():
        return mp.cpu_count()
