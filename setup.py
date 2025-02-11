"""DeepJR setup file.

Authors:
Deepa Tilwani <dtilwani@mailbox.sc.edu>
Christian O'Reilly <christian.oreilly@sc.edu>
License: MIT
"""

from pathlib import Path
from setuptools import setup, find_packages

with Path('requirements.txt').open() as f:
    requirements = f.read().splitlines()

extras = {}

extras_require = {}
for extra, req_file in extras.items():
    with Path(req_file).open() as file:
        requirements_extra = file.read().splitlines()
    extras_require[extra] = requirements_extra

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='deepjr',
    version='0.1.0',
    description='Deep Jansen-Rit inference',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Deepa Tilwani",
    author_email='dtilwani@mailbox.sc.edu',
    url='https://github.com/lina-usc/Jansen-Rit-Model-Benchmarking-Deep-Learning',
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
    include_package_data=True,
)
