from setuptools import setup, find_packages


def get_long_description():
    with open("README.md", "r") as readme_file:
        return readme_file.read()


setup(
    name='mechamodlearn',
    description='PyTorch framework for learning mechanical systems',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    install_requires=[
        'click',
        'py-dateutil',
        'ipdb',
        'torch >= 1.0',
        'numpy >= 1.13',
        'matplotlib',
        'more_itertools',
        'scipy',
        'tqdm',
        'numba >= 0.37',
        'termcolor',
        'tensorboardX',
    ],
    url='https://github.com/sisl/mechamodlearn/',
    packages=find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',),
    version='0.0.1',
    author='kunalmenda, rejuvyesh',
    author_email='kmenda@stanford.edu, mail@rejuvyesh.com',
    license='MIT',)
