1. Create virtual env

python -m venv py36-venv

2. Copy everything from "Code" directory to "py36-venv" directory

3. Change dir then activate venv

cd .\py36-venv\
.\Scripts\activate

4. Install pip package from requirements.txt

python -m pip install -r .\requirements.txt

5. Copy Lib from custom_lib then overwrite the Lib

6. Build exe with pyinstaller and main.spec

pyinstaller -F --clean --onefile .\main.spec