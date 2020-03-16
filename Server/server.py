from flask import Flask, render_template, url_for, jsonify, request, send_from_directory
import requests

app = Flask(__name__)


@app.route("/search")
@app.route("/")
def search():
    return render_template('search.html')


@app.route("/ajax", methods = ['POST', 'GET'])
def ajax():
    print(dict(request.values))
    response = requests.get('https://monkagiga.firebaseio.com/.json')
    return jsonify(response.text)


if __name__ == '__main__':
    app.run(debug = True)
