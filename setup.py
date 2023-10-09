import setuptools
import pathlib


setuptools.setup(
    name='smartplay',
    version='0.0.1',
    description='A benchmarking tool for LLMs with games',
    url='',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    # package_data={'crafter': ['data.yaml', 'assets/*']},
    entry_points={'console_scripts': ['smartplay=smartplay.run_gui:main']},
    install_requires=[
        'numpy', 
        'pandas',
        'pygame', 
        'connected-components-3d', 
        'gym', 
        'minedojo', 
        'imageio', 
        'pillow', 
        'opensimplex', 
        'ruamel.yaml',
        'importlib-metadata==6.6.0',
        'importlib-resources==5.12.0',
        'vgdl @ git+https://github.com/ahjwang/py-vgdl',
    ],
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
