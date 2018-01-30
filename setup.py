from setuptools import setup, find_packages

__version__ = "unknown"

# "import" __version__
for line in open("sfs/__init__.py"):
    if line.startswith("__version__"):
        exec(line)
        break

setup(
    name="sfs",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'numpy!=1.11.0',  # https://github.com/sfstoolbox/sfs-python/issues/11
        'scipy',
    ],
    author="SFS Toolbox Developers",
    author_email="sfstoolbox@gmail.com",
    description="Sound Field Synthesis Toolbox",
    long_description=open('README.rst').read(),
    license="MIT",
    keywords="audio SFS WFS Ambisonics".split(),
    url="http://github.com/sfstoolbox/",
    platforms='any',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
    zip_safe=True,
)
