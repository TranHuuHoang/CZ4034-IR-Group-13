from elasticsearch import Elasticsearch

import Basketball
import os
import sys
import requests


# TAG Definition
TAG = "[{}]".format(os.path.basename(__file__))

# Variable Definition
basketball_data = "Raw_Data/basketball.xlsx"

INITIAL_RETRIES = 0
MAX_RETRIES = 3
DECODING_STANDARD = "utf-8"

def run():
    """
    Driver class for the search engine
    """

    print("{}: Get Elasticsearch connection".format(TAG))
    elasticsearch_connection = get_elasticsearch_connection()

    print("{}: Get Basketball Dataframe".format(TAG))
    basketball = Basketball.get_data(basketball_data)

    print("{}: Index basketball_data into elasticsearch".format(TAG))
    Basketball.index_data_in_elasticsearch(basketball, elasticsearch_connection)


def get_elasticsearch_connection():

    elasticsearch_retries = INITIAL_RETRIES
    elasticsearch_max_retries = MAX_RETRIES

    elasticsearch_host = 'localhost'

    elasticsearch_port = 9200

    print("{}: Test if the elasticsearch cluster is reachable".format(TAG))
    elasticsearch_url = 'http://{}:{}'.format(
        elasticsearch_host,
        elasticsearch_port
    )

    while(True):
        try:
            response = requests.get(elasticsearch_url)
            break
        except Exception as e:
            if elasticsearch_retries == elasticsearch_max_retries:
                print(
                    "{}: Test Elasticsearch reachability exceeded max retries"
                    .format(TAG)
                )
                print(e)
                sys.exit(1)
            else:
                elasticsearch_retries = elasticsearch_retries + 1
                print(
                    "{}: Elasticsearch test reachability retries: {}".format(
                        TAG,
                        elasticsearch_retries
                    )
                )

    # Assert that response from requests is OK
    assert response.status_code is requests.codes.ok
    print("{}: Elasticsearch cluster reachable".format(TAG))

    elasticsearch_connection = Elasticsearch(
        [
            {
                'host': elasticsearch_host,
                'port': elasticsearch_port
            }
        ]
    )

    return elasticsearch_connection


if __name__ == '__main__':
    run()