import numpy as np
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)

class Buffer(object):
    """
    """
    def __init__(self, nch=None, nsamp=None):
        self._nch = 1 if nch is None else nch
        self._nsamp = 2**10 if nsamp is None else nsamp
        self._dat = np.zeros((self.nch, self.nsamp+1), dtype=np.float32)

        self._writehead = 0
        self._readhead  = 0

        self.allow_overflow = False
        self.overflowed_samples = 0

    def to_write(self):
        """
        Returns the amount of free space currently in the buffer, i.e. the
        number of samples that can be written.
        """
        diff = self.readhead - self.writehead - 1
        if self.readhead > self.writehead:
            return diff
        else:
            return diff + self.nsamp

    def to_read(self):
        """
        Returns the number of samples pending in the buffer, i.e. available
        but not yet read.
        """
        diff = self.writehead - self.readhead
        if self.writehead >= self.readhead:
            return diff
        else:
            return diff + self.nsamp

    def write(self, x, axis=-1):
        """
        Writes signal packet <x>, a channels-by-samples numpy array, to the
        buffer.
        """
        available = self.to_write()

        x = np.asarray(x).view()
        x = np.atleast_2d(x) if x.ndim < 2 else x

        nch, nsamp = x.shape
        if nch != self.nch:
            raise ValueError("Incoming data has the wrong number of channels.")

        if nsamp > available:
            self.overflowed_samples += nsamp - available
            if self.allow_overflow:
                nsamp = available
                x = x[:,:nsamp]
            else:
                raise RuntimeError("ring buffer overflow")

        n = min([nsamp, self.nsamp - self.writehead])
        m = max([0, nsamp - n])

        self._dat[:,self.writehead:self.writehead+n] = x[:,:n]
        self._writehead = (self.writehead + n) % self.nsamp
        self._writehead += m
        self._dat[:, :m] = x[:, n:n+m]

    def read(self, nsamp=None, remove=True):
        """
        Reads <nsamp> samples from the buffer, returning a channels-by-samples
        numpy array. By default, reading removes these samples from the buffer:
        set <remove> to False if you want to prevent this.
        """###
        available = self.to_read()

        nsamp = available if nsamp is None else nsamp

        if nsamp > available:
            raise RuntimeError("ring buffer underflow")

        x = numpy.zeros((self.nch, nsamp), dtype=self._dat.dtype)
        n = min([nsamp, self.nsamp - self.readhead])
        x[:,:n] = self._dat[:,self.readhead:self.readhead+n]

        m = max(0, nsamp - n)
        x[:,n:n+m] = self._dat[:,:m]

        if remove:
            self._readhead = ((self.readhead + n) % self.samples()) + m

        return x

    def forget(self, nsamp):
        nsamp = min([nsamp, self.to_read()])
        self._readhead = (self.readhead + nsamp) % self.samples()

    @property
    def nch(self):
        return self._nch

    @property
    def nsamp(self):
        return self._nsamp

    @property
    def buffer(self):
        return

    @property
    def readhead(self):
        return self._readhead

    @property
    def writehead(self):
        return self._writehead
