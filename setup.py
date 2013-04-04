try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
        'description': 'Gravity currents in two layer fluid: processing of lab data',
        'author': "Aaron O'Leary",
        'url': 'http://github.com/aaren/lab-waves',
        'download_url': 'http://github.com/aaren/lab-waves/download',
        'author_email': 'eeaol@leeds.ac.uk',
        'version': '0.1',
        'install_requires': ['nose'],
        'packages': ['labwaves'],
        'scripts': [],
        'name': 'labwaves'
        }

setup(**config)
