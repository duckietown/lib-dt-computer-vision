from setuptools import find_packages, setup

# :==> Fill in your project data here
# The package name is the name on PyPI
# it is not the python module names.
package_name = "dt-computer-vision"
library_webpage = f"http://github.com/duckietown/lib-{package_name}"
maintainer = "Andrea F. Daniele"
maintainer_email = "afdaniele@duckietown.org"
short_description = "Computer Vision components of Duckietown's autonomy behavior."
full_description = """
Computer Vision components of the autonomous behavior pipeline running on Duckietown robots.
"""

# Read version from the __init__ file
def get_version_from_source(filename):
    import ast

    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith("__version__"):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError("No version found in %r." % filename)
    if version is None:
        raise ValueError(filename)
    return version


version = get_version_from_source(f"src/{package_name.replace('-', '_')}/__init__.py")

install_requires = [
    # opencv4
    "opencv-python-headless<=4.8.1.78",
    # numpy
    "numpy<=1.26.2"
]
tests_require = []

# compile description
underline = "=" * (len(package_name) + len(short_description) + 2)
description = """
{name}: {short}
{underline}

{long}
""".format(
    name=package_name,
    short=short_description,
    long=full_description,
    underline=underline,
)

packages = find_packages("./src")

print("The following packages were found:\n\t - " + "\n\t - ".join(packages) + "\n")

# setup package
setup(
    name=f"lib-{package_name}",
    author=maintainer,
    author_email=maintainer_email,
    url=library_webpage,
    tests_require=tests_require,
    install_requires=install_requires,
    package_dir={"": "src"},
    packages=find_packages("./src"),
    include_package_data=True,
    long_description=description,
    version=version,
)
