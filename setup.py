import setuptools
from glob import glob

# Will load the README.md file into a long_description of the package
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
# Load the requirements file
with open('requirements.txt') as f:
    required = f.read().splitlines()
if __name__ == "__main__":
    setuptools.setup(
        name='MM2SF',
        version='1.0',
        author='Miguel Gallegos',
        author_email='gallegosmiguel@uniovi.es',
        description="An Unsupervised Machine Learning Approach for the Automatic Construction of Local Chemical Descriptors",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/m-gallegos/MM2SF',
        project_urls = {
            "MM2SF": "https://github.com/m-gallegos/MM2SF"
        },
        license='MIT',
        python_requires='>=3.7.3, <=3.11.5',
        install_requires=required,
        zip_safe= False,
        package_dir={"": "src"},
        packages=setuptools.find_packages(where='src'),
        include_package_data=True,
    )
