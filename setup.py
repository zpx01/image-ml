from setuptools import find_packages, setup
setup(
    name="image-ml",
    packages=find_packages(include=["image-ml"]),
    version="0.1.0",
    description="Image Processing with Machine Learning",
    author="Zeeshan Patel, Kushaagra Gupta, Brendan Mai, Aryan Dua",
    license="MIT",
    install_requires=["numpy", "scipy", "scikit-image"],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)