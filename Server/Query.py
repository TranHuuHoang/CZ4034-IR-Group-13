import os
import json


# TAG Definition
TAG = "[{}]".format(os.path.basename(__file__))

# Variable Definition
MISSING_VALUE = "-1"
QUERY_RETURN_SIZE = 5
INDEX_NAME = "basketball"


def query(term, elasticsearch_connection):
    """
    Get list of recommended tweet for each user input

    Parameters
    ----------
    term : string
        String entered by user to perform search
    elasticsearch_connection : Elasticsearch Connection
        An Elasticsearch Connection

    Returns
    -------
    dict
        key - Search ID
        value - list of candidate recommended for the query
    """

    recommendation = {}

    print("{}: Query Elasticsearch for recommendation".format(TAG))

    id = 1

    try:

        response = search_elasticsearch(elasticsearch_connection, term)

        result = response['hits']['hits']

        extracted_result = extract_result(result)

        recommendation = extracted_result

    except Exception as e:
        print("{}: {}".format(TAG, e))

    print("{}: Query completed".format(TAG))

    return recommendation


def search_elasticsearch(elasticsearch_connection, term):
    """
    Query Elasticsearch cluster for list of recommended candidate

    Parameters
    ----------
    elasticsearch_connection : Elasticsearch Connection
        An Elasticsearch connection
    term : string
        String entered by user to perform search

    Returns
    -------
    dict
        Elasticsearch Response
    """

    search_body = get_search_body(term)

    response = elasticsearch_connection.search(
        index=INDEX_NAME,
        body=search_body
    )

    return response


def get_search_body(term):
    """
    To get the should body of an Elasticsearch query in Elasticsearch DSL

    Parameters
    ----------
    term : string
        String entered by user to perform search

    Returns
    -------
    json
        Search body of an Elasticsearch query
    """

    # body = {
    #     "query" : {
    #         "term" : { "TweetText" : term}
    #     }
    # }

    # return body

    body = {}
    body['query'] = {}
    body['query']['bool'] = {}
    bool_body = body['query']['bool']

    body['size'] = QUERY_RETURN_SIZE
    bool_body['should'] = {}
    bool_body['should']['match'] = {}
    bool_body['should']['match']["TweetText"] = {
                "query": term
            }

    return body


def extract_result(result):
    """
    Extract ID and ranking score from Elasticsearch result

    Parameters
    ----------
    result : json
        Elasticsearch query response

    Returns
    -------
    list
        A list of recommended tweets in decreasing order of suitability
    """

    extracted_result = []

    for tweet in result:
        segment_result = []

        # tweet_id = tweet['_id']
        # tweet_score = float(tweet['_score'])

        # segment_result.append(tweet_id)
        # segment_result.append(tweet_score)

        tweet_source = tweet['_source']

        # segment_result.append(tweet_source)

        extracted_result.append(tweet_source)

    return extracted_result
