## Running the project

- Clone the project
- Create new python environment by running command in project root directory:
```
python -m venv env
```
- Activate the newly created environment:
```
.\env\Scripts\activate
```

- Run the following command in root directory to install all packages:
```
pip install -r requirements.txt
```
- Run the project by running command:
```
uvicorn main:app --reload
```