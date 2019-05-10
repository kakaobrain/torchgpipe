from setuptools import setup


about = {}  # type: ignore
with open('torchgpipe/__about__.py') as f:
    exec(f.read(), about)  # pylint: disable=W0122


setup(
    name='torchgpipe',
    version=about['__version__'],
    author='Kakao Brain',
    maintainer='Heungsub Lee, Myungryong Jeong',
    zip_safe=False,
    packages=['torchgpipe'],
    install_requires=['torch'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
