import tweepy as tw
import pandas as pd
import time

consumer_key = "YOUR API KEY HERE"
consumer_secret = "YOUR API KEY HERE"
access_token = "YOUR API KEY HERE"
access_token_secret = "YOUR API KEY HERE"

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify = True)

def scrape(query, count):
    cur = tw.Cursor(api.search, q=query+"-filter:retweets", lang="en")
    tweet_info = [(tweet.id_str, tweet.created_at, tweet.text, tweet.user.name, "@"+tweet.user.screen_name, str(tweet.in_reply_to_status_id)) for tweet in cur.items(count)]
    df = pd.DataFrame(tweet_info, columns = ["TweetID", "Date(SGT - 9)", "TweetText", "AuthorName", "UserHandle", "ReplyTweetID"])
    return df


##df = scrape("soccer", 2500)
##df.to_excel("output.xlsx", index = False)


