import os
import sys
import pandas as pd


# TAG Definition
TAG = "[{}]".format(os.path.basename(__file__))

# Variable Definition
INITIAL_RETRIES = 0
MAX_RETRIES = 3
MISSING_VALUE = "-1"

# SPLIT_BULK_DATA_PART_SIZE must be divisible by 2
SPLIT_BULK_DATA_PART_SIZE = 500


def get_data(filename):
    """
    Return Data for indexing into Elasticsearch

    Parameters
    ----------
    filename: string
        File name from where the data are extracted from

    Returns
    -------
    Pandas Dataframe
        A pandas dataframe containing data information
    """

    data = pd.read_excel(filename)
    
    return data


def index_data_in_elasticsearch(data, elasticsearch_connection):
    """
    Index data information into elasticsearch

    Parameters
    ----------
    data : Pandas Dataframe
        Dataframe containing data attributes

    elasticsearch_connection : Elasticsearch Connection
        An Elasticsearch connection
    """

    data_index_name = "basketball"

    data_index_configuration = {
        "mappings": {
            "properties": {
                "TweetID": {
                    "type": "keyword"
                    },
                "Date": {
                    "type": "date" 
                    # "format": "yyyy-MM-dd HH:mm:ss"
                    },
                "TweetText": {
                    "type": "text"
                    },
                "AuthorName": {
                    "type": "text"
                    },
                "UserHandle": {
                    "type": "text"
                    },
                # If blank, string "None" is put in
                "ReplyTweetID": {
                    "type": "keyword"
                    }
            }
        }
    }

    create_index(
        elasticsearch_connection,
        data_index_name,
        data_index_configuration
    )

    bulk_data = prepare_bulk_data_for_indexing(data, data_index_name)

    splitted_bulk_data = split_bulk_data(bulk_data)

    insertion_index = 1
    for split_part in splitted_bulk_data:
        print(
            "{}: Inserting part {} of {}".format(
                TAG,
                insertion_index,
                len(splitted_bulk_data)
            )
        )
        bulk_index(split_part, elasticsearch_connection, data_index_name)
        insertion_index = insertion_index + 1


def create_index(elasticsearch_connection, index_name, index_configuration):
    """
    Create index in elasticsearch with
    the given index_name and index_configuration

    Parameters
    ----------
    elasticsearch_connection : elasticsearch.Elasticsearch
        Elasticsearch low-level client
    index_name : str
        Name of index to create
    index_configuration : dict
        Configuration of index in Elasticsearch DSL
    """

    if elasticsearch_connection.indices.exists(index_name):
        print(
            "{}: Index with name '{}' already exists".format(TAG, index_name)
        )
        print("{}: Deleting existing index '{}'".format(TAG, index_name))
        response = elasticsearch_connection.indices.delete(index=index_name)
        assert response['acknowledged'] is True

    create_index_retries = INITIAL_RETRIES
    create_index_max_retries = MAX_RETRIES

    while(True):
        try:
            print("{}: Creating index '{}'".format(TAG, index_name))
            response = elasticsearch_connection.indices.create(
                index=index_name,
                body=index_configuration
            )
            assert response['acknowledged'] is True
            print("{}: Index '{}' created".format(TAG, index_name))
            break
        except Exception as e:
            if create_index_retries is create_index_max_retries:
                print("{}: Create index exceeded max retries".format(TAG))
                print(e)
                sys.exit(1)
            else:
                create_index_retries = create_index_retries + 1
                print(
                    "{}: Create index retries: {}".format(
                        TAG,
                        create_index_retries
                    )
                )


def prepare_bulk_data_for_indexing(data, data_index_name):
    """
    Prepare data dataframe data into Elasticsearch DSL format
    for bulk indexing

    Parameters
    ----------
    data : Pandas Dataframe
        Dataframe containing data attributes
    data_index_name : str
        Name of index to index document to

    Returns
    -------
    list
        A list of data data in Elasticsearch DSL for indexing
    """

    bulk_data = []

    print("{}: Preparing bulk data for indexing".format(TAG))

    for index, row in data.iterrows():

        TweetID = row['TweetID']
        Date = row['Date(SGT - 9)']
        TweetText = row['TweetText']
        AuthorName = row['AuthorName']
        UserHandle = row['UserHandle']
        ReplyTweetID = row['ReplyTweetID']

        action = {
            "index": {
                "_index": data_index_name,
                # "_type": data_index_name,
                "_id": TweetID
            }
        }

        data_impl = {
            "TweetID": TweetID,
            "Date": Date,
            "TweetText": TweetText,
            "AuthorName": AuthorName,
            "UserHandle": UserHandle,
            "ReplyTweetID": ReplyTweetID
        }

        bulk_data.append(action)
        bulk_data.append(data_impl)

    print("{}: {} entries prepared in bulk_data".format(TAG, len(bulk_data)/2))
    
    return bulk_data


def split_bulk_data(bulk_data):
    """
    Split bulk data into smaller chunk
    so as not to overwhelm Elasticsearch cluster

    Parameters
    ----------
    bulk_data : list
        list containing `action` and `data` json written in Elasticsearch DSL

    Returns
    -------
    list
        A list containing smaller list of bulk_data with size determined by
        SPLIT_BULK_DATA_PART_SIZE
    """

    print(
        "{}: Splitting bulk data into parts of {}".format(
            TAG, SPLIT_BULK_DATA_PART_SIZE
        )
    )
    splitted_bulk_data = []

    for i in range(0, len(bulk_data), SPLIT_BULK_DATA_PART_SIZE):
        splitted_bulk_data.append(bulk_data[i:i+SPLIT_BULK_DATA_PART_SIZE])

    return splitted_bulk_data


def bulk_index(bulk_data, elasticsearch_connection, data_index_name):
    """
    Bulk insert document into Elasticsearch

    Parameters
    ----------
    bulk_data : list
        A list containing `action` and `data` json written in Elasticsearch DSL
    elasticsearch_connection : Elasticsearch Connection
        An Elasticsearch connection
    data_index_name : str
        Elasticsearch index to use for insertion
    """

    bulk_index_retries = INITIAL_RETRIES
    bulk_index_max_retries = MAX_RETRIES

    while(True):
        try:
            print("{}: Bulk indexing data document".format(TAG))
            elasticsearch_connection.bulk(
                index=data_index_name,
                body=bulk_data,
                refresh=(True)
            )
            break
        except Exception as e:
            if bulk_index_retries == bulk_index_max_retries:
                print("{}: Bulk indexing exceeded max retries".format(TAG))
                print(e)
                sys.exit(1)
            else:
                bulk_index_retries = bulk_index_retries + 1
                print(
                    "{}: Bulk indexing retries: {}".format(
                        TAG,
                        bulk_index_retries
                    )
                )
