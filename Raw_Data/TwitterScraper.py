import tweepy as tw
import pandas as pd
import time

consumer_key = "VadtVXSmHFSAGJMOehPUBlEwY"
consumer_secret = "fT7bVc7d4qU0G7EIddUIVHh6mdGCAsa3vyLxHCwbMGVApN2h4C"
access_token = "2580718434-CdBur2K08VRxqp9IcOLey0TdryjRsV6IgSr4uZr"
access_token_secret = "lWYkgidOtYXGGM5puXaxIbnmyKN7iMQezVDgh0Gp4ToFf"

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify = True)

def scrape(query, count):
    cur = tw.Cursor(api.search, q=query+"-filter:retweets", lang="en", tweet_mode = "extended")
    tweet_info = [(tweet.id_str, tweet.created_at, tweet.full_text, tweet.user.name, "@"+tweet.user.screen_name, str(tweet.in_reply_to_status_id)) for tweet in cur.items(count)]
    df = pd.DataFrame(tweet_info, columns = ["TweetID", "Date(SGT - 9)", "TweetText", "AuthorName", "UserHandle", "ReplyTweetID"])
    return df


##df = scrape("soccer", 2500)
##df.to_excel("output.xlsx", index = False)


