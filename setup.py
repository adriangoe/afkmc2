from distutils.core import setup
setup(
  name = 'afkmc2',
  packages = ['afkmc2'],
  version = '0.1',
  description = 'Assumption Free and Efficient K-Means Seeding',
  author = 'Adrian Goedeckemeyer',
  author_email = 'adrian@minerva.kgi.edu',
  url = 'https://github.com/adriangoe/afkmc2',
  download_url = 'https://github.com/adriangoe/afkmc2/archive/0.1.tar.gz',
  keywords = ['kmeans', 'seeding', 'sklearn', 'numpy'],
  classifiers = [],
  install_requires=[
      "numpy",
  ],
)