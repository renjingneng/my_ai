from os.path import join, dirname
from setuptools import setup, find_packages


def read_file_content(filepath):
    with open(join(dirname(__file__), filepath), encoding="utf8") as fp:
        return fp.read()


long_description = read_file_content('README.md')

setup(name='my_ai',
      version='0.0.1',
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
