from os.path import join, dirname
from setuptools import setup, find_packages
import re


def read_file_content(filepath):
    with open(join(dirname(__file__), filepath), encoding="utf8") as fp:
        return fp.read()


def find_version(filepath):
    content = read_file_content(filepath)
    # re.M means re.MULTILINE :https://docs.python.org/3/library/re.html#re.M
    version_match = re.search(r'^__version__ = [\'"]([^\'"]+)[\'"]', content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


long_description = read_file_content('README.md')
version = find_version(join('my_ai', '__init__.py'))

setup(name='my_ai',
      version=version,
      url='https://github.com/renjingneng/my_ai',
      author='Jingneng Ren',
      author_email='renjingneng@gmail.com',
      description='my ai journey',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT',

      packages=find_packages(exclude=('test', 'doc', 'example')),
      include_package_data=True,
      keywords=[
          "deep learning", "neural network", "machine learning"]
      )
