"""sophon python module
"""
__all__ = []

try:
    import numpy
except ImportError:
    print('SAIL bindings requires "numpy" package.')
    print('Install it via command:')
    print('   pip3 install numpy')
    raise