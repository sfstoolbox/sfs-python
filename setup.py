from setuptools import setup

exec(open('sfs/_version.py').read())

setup(
    name="SoundFieldSynthesis",
    version=__version__,
    packages=['sfs'],

    author="SFS Toolbox Developers",
    author_email="sfstoolbox@gmail.com",
    description="Sound Field Synthesis Toolbox",
    long_description=open('README.rst').read(),
    license="MIT",
    keywords="audio SFS WFS Ambisonics".split(),
    url="http://github.com/sfstoolbox",
    download_url="http://github.com/sfstoolbox/sfs-python/releases",
    platforms='any',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
)
