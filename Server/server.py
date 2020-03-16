from flask import Flask, render_template, url_for, jsonify, request, send_from_directory
import requests

import Query
import Elastic

app = Flask(__name__)


@app.route("/home")
@app.route("/")
def home():
    return render_template('search.html')


@app.route("/ajax", methods = ['POST', 'GET'])
def ajax():
    print(dict(request.values))
    # response = requests.get('https://monkagiga.firebaseio.com/.json')

    elasticsearch_connection = Elastic.get_elasticsearch_connection()
    term = request.args.get('query')
    
    response = Query.query(term, elasticsearch_connection)
    print(response)
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug = True)
