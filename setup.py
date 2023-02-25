from setuptools import setup, find_packages


setup(
   name='language_engine',
   version='0.0.1',
   packages=find_packages(
      exclude=['examples', 'scripts'], 
   ),
   install_requires=[
            # TODO: add other dependencies
            "gdown",
            "evaluate",
            "scikit-learn",
            "datasets",
            "protobuf",
            "accelerate",
            "torch",
            "tokenizers",
            "datasets",
            "hydra",
            "transformers"
   ],
)