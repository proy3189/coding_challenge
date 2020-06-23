from os.path import realpath, dirname, join

from setuptools import setup, find_packages

DISTNAME = 'talpa'
DESCRIPTION = 'Reproduce activities from sensor data recieved from roofbolter '
MAINTAINER = 'proy'
MAINTAINER_EMAIL = 'roypriyanka3103@gmail.com'
VERSION = "1.0"
LICENSE = "Apache"
DOWNLOAD_URL = "https://github.com/proy3189/coding_challenge"
PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

if __name__ == "__main__":
    setup(name=DISTNAME,
          version=VERSION,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          packages=find_packages(),
          install_requires=install_reqs,
          url=DOWNLOAD_URL,
          include_package_data=True)
