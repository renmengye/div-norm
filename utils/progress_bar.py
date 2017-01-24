"""
A python progress bar.

Example 1:
    N = 1000
    pb = progress_bar.get(N)
    
    for i in xrange(N):
        do_something(i)
        pb.increment()

Example 2:
    N = 1000
    for i in progress_bar.get(N):
        do_something(i)

Example 3:
    l = ['a', 'b', 'c']
    for c in progress_bar.get_iter(l):
        do_something(c)
"""


from __future__ import division
import sys

def get(length):
    """Returns a ProgressBar object.

    Args:
        length: number, total number of objects to count.
    """
    return ProgressBar(length)


def get_iter(iterable):
    """Returns a progress bar from iterable."""
    return ProgressBar(len(iterable), iterable=iter(iterable))


class ProgressBar(object):
    """Prints a dotted line in a standard terminal."""

    def __init__(self, length, iterable=None, width=70):
        """Constructs a ProgressBar object.

        Args:
            length: number, total number of objects to count.
            width: number, width of the progress bar.
        """
        self.length = length
        self.value = 0
        self.progress = 0
        if iterable is None:
            self.iterable = iter(range(length))
        else:
            self.iterable = iterable
        self.width = width
        self._finished = False
        pass

    def __iter__(self):
        """Get iterable object."""
        return self

    def next(self):
        """Iterate next."""
        self.increment()
        return self.iterable.next()

    def increment(self, value=1):
        """Increments the progress bar.

        Args:
            value: number, value to be incremented, default 1.
        """
        self.value += value
        while not self._finished and \
              self.value / self.length > self.progress / self.width:
            sys.stdout.write('.')
            sys.stdout.flush()
            self.progress = self.progress + 1
        if self.progress == self.width and not self._finished:
            sys.stdout.write('\n')
            sys.stdout.flush()
            self._finished = True
        pass
