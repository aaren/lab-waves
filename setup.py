try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
        'description': 'My Project',
        'author': "Aaron O'Leary",
        'url': 'http://github.com/aaren/NAME',
        'download_url': 'http://github.com/aaren/NAME/download',
        'author_email': 'eeaol@leeds.ac.uk',
        'version': '0.1',
        'install_requires': ['nose'],
        'packages': ['NAME'],
        'scripts': [],
        'name': 'NAME'
        }

setup(**config)
