from flask import Flask, render_template, url_for, jsonify, request
import requests

import Query
import Elastic
import Health

app = Flask(__name__)

result = None
elasticsearch_connection = None

@app.route("/home")
@app.route("/")
def home():
    return render_template('search.html')


@app.route("/ajax", methods = ['POST', 'GET'])
def ajax():
    global result
    global elasticsearch_connection

    print(dict(request.values))
    # response = requests.get('https://monkagiga.firebaseio.com/.json')

    elasticsearch_connection = Elastic.get_elasticsearch_connection()
    term = request.args.get('query')
    
    result = Query.query(term, elasticsearch_connection)
    response = result
    print(response)
    return jsonify(response)


@app.route('/feedback', methods = ['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        data = request.form.getlist('key')
        id = result[int(data[0])]["TweetID"]
        Health.updateFeedback(elasticsearch_connection, id)

    return "Disliked"


if __name__ == '__main__':
    app.run(debug = True)
