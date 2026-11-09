from pathlib import Path
from setuptools import find_packages, setup

root = Path(__file__).parent
requirements = [line.strip() for line in (root/'requirements.txt').read_text().splitlines()
                if line.strip() and not line.startswith('#')]

setup(
    name='rps-table',
    version='1.0.0',
    description='RPS semi-supervised table structure recognition',
    long_description=(root/'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={'rps':['vocab/*.txt']},
    python_requires='>=3.8,<3.10',
    install_requires=requirements,
)
