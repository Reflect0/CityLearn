from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A grid integrated version of the CityLearn project'
LONG_DESCRIPTION = 'A grid integrated version of the CityLearn project'

# Setting up
setup(
        name="citylearn",
        version=VERSION,
        author="Jose Vazquez Canteli edited by Aisling Pigott",
        author_email="aisling.pigott@colorado.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that
        # needs to be installed along with your package. Eg: 'caer'
        classifiers= []
)
