from setuptools import setup,find_packages
from typing import List
hypen="-e ."
def get_requirements(file_path:str)-> List[str]:
    with open(file_path,"r") as file:
        requirements=file.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if hypen in requirements:
            requirements.remove(hypen)
    return requirements

setup(name="Mushroom--project",
author="Saeed",
version="0.0.1",
author_email="saidshaikh.nagar@gmail.com"
,packages=find_packages(),
install_requires=get_requirements("requirements.txt"))