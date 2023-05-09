from setuptools import setup, find_packages

setup(
    name="coopreaching",
    version="0.0.1",
    description="Cooperative Reaching Environment",
    install_requires=["numpy", "gym>=0.14.0", "pyglet"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
