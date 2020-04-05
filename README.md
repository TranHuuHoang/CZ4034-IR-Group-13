# CZ4034-IR-Group-13

## Pre-requisite

- Must have python 3.6+

- Must have pip

- Install dependencies

  - `pip install numpy pandas flask requests elasticsearch`

## Instruction to run

- Start elasticsearch server

  - cd to the project directory

  - Run `chmod +x elasticsearch.sh`

  - Run `./elasticsearch.sh`

- Setup web server

  - Open new terminal

  - Run `python Elastic.py && python server.py`

  - Go to `http://127.0.0.1:5000/`
