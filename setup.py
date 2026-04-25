from setuptools import setup, find_packages

setup(
    name="itimecontrol-assistant",
    version="0.1.0",
    description="Asistente conversacional inteligente basado en fine-tuning y RAG para iTimeControl",
    author="Tu Nombre",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
