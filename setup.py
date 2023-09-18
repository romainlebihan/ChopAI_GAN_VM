from setuptools import setup
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]
setup(name='ChopAI', description = "chop AI", packages = [], install_requires = requirements)
