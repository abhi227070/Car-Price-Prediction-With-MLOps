from setuptools import find_packages, setup

def get_requirements(file_path):
    
    requirements = []
    
    with open(file_path) as file_obj:
        
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        
    return requirements


setup(
    name="Car Price Prediction",
    version='0.0.1',
    description= 'Predict the present price of the car from the preview history of it.',
    author= 'abhijeet',
    author_email= 'abhijeetmaharana77@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)