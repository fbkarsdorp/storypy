import setuptools
from distutils.core import setup 

setup(
    name='storypy',
    version='0.0.1',
    packages=['storypy'],
    url='http://fbkarsdorp.github.io/storypy',
    author='Folgert Karsdorp',
    author_email='fbkarsdorp AT fastmail DOT nl',
    install_requires=['numpy', 
                      'pandas', 
                      'networkx', 
                      'distance', 
                      'scikit-learn', 
                      'pattern', 
                      'dtw'],
    dependency_links=[
        "git+https://github.com/fbkarsdorp/dtw.git"])
