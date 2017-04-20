from setuptools import setup

setup(
    name = 'afkmc2',
    packages = ['afkmc2'],
    version = '0.1',
    description = 'Assumption Free and Efficient K-Means Seeding',
    author = 'Adrian Goedeckemeyer',
    author_email = 'adrian+pypi@minerva.kgi.edu',
    url = 'http://afkmc2.readthedocs.io/en/latest/index.html',
    download_url = 'https://github.com/adriangoe/afkmc2/archive/0.1.tar.gz',
    keywords = ['kmeans', 'seeding', 'sklearn', 'numpy'],
    classifiers = [],
    install_requires=[
        "numpy",
    ],
)