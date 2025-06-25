from setuptools import find_packages,setup

hpe='-e .'
def get_requirements(file_path:str)->list[str]:
    '''
    returns list of requirements
    '''
    requirements=[]
    with open(file_path)as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]
        if hpe in requirements:
            requirements.remove(hpe)
    return requirements

setup(
    name='EtoEproject',
    version='0.0.1',
    author='Aryaman Prasad',
    author_email='aryamanips@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)