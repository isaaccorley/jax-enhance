from setuptools import setup, find_packages

__version__ = "0.0.1"
url = "https://github.com/isaaccorley/jax-enhance"

with open("requirements.txt", "r") as f:
    install_requires = f.read().strip().splitlines()

setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov', 'mock']

setup(
    name='jax_enhance',
    packages=find_packages(exclude=['examples']),
    version=__version__,
    license='Apache License 2.0',
    description='Image Super-Resolution Library for Jax',
    author='Isaac Corley',
    author_email='isaac.corley@my.utsa.edu',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'jax',
        'image-super-resolution',
        'computer-vision',
        'deep-neural-networks',
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)