# CZ4034-IR-Group-13

## Pre-requisite

- Must have python 3.6+

- Must have pip

- Install dependencies

  - `pip install numpy pandas flask requests elasticsearch`

## Instruction to run

- Start elasticsearch server (If you used Linux machine)

  - cd to the project directory

  - Run `chmod +x elasticsearch.sh`

  - Run `./elasticsearch.sh`

- If you are not using Linux, please download elasticsearch from the link below

  `https://www.elastic.co/downloads/elasticsearch`

  - Unzip the file to a favorite folder, and navigate to that folder

  - For Window, run: `.\bin\elasticsearch.bat`

  - For MacOS, run: `bin/elasticsearch`

- Setup web server

  - Open new terminal

  - Run `python Elastic.py && python server.py`

  - Go to `http://127.0.0.1:5000/`
