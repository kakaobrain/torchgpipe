from setuptools import setup


about = {}  # type: ignore
with open('torchgpipe/__version__.py') as f:
    exec(f.read(), about)  # pylint: disable=W0122
version = about['__version__']
del about


setup(
    name='torchgpipe',

    version=version,

    license='Apache License 2.0',
    url='https://github.com/kakaobrain/torchgpipe',
    author='Kakao Brain',
    maintainer='Heungsub Lee, Myungryong Jeong',

    description='GPipe for PyTorch',
    keywords='pytorch gpipe',

    zip_safe=False,

    packages=['torchgpipe'],
    package_data={'torchgpipe': ['py.typed']},

    install_requires=['torch>=1,<1.1'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest>=4'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Typing :: Typed',
    ],
)
