from setuptools import find_namespace_packages, setup


setup(packages=["vision_encoder_eval"] + [
    "vision_encoder_eval." + name
    for name in find_namespace_packages("src", exclude=["__pycache__", "*.__pycache__"])
])
