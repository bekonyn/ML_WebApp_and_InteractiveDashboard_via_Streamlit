import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# st.title("Hello, World!")
# # u can use ## to change markdown lvl
# st.markdown("## My first streamlit dashboard!")

st.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets about US Airlines")

st.markdown("This application is  a Streamlit dashboard to analyze the sentiment of Tweets🐦")
st.sidebar.markdown("This application is a Streamlit dashboard to analyze the sentiment of Tweets🐦")

data_url=("Tweets.csv")

@st.cache(persist=True)
def load_data():
    data=pd.read_csv(data_url)
    data["tweet_created"]=pd.to_datetime(data["tweet_created"])
    return data

data=load_data()
#st.cache is a decorator (func decorator) it ll help our webapp that unless u change sth in load_data func
#it ll only be loaded once. and then its stored on cache to recall again. so yr webapp dont need to rerun all
#that df again.

#u can see yr data if u want right now, but we ll add more stuff so i commented out this part.
#st.write(data)

#we dont always want to see our raw_data table. lets add a sidebar that ll help us.sidebar w radio button
#we add 1 sidebar and 1 "radio button widget"
st.sidebar.subheader("Show random tweet")
random_tweet=st.sidebar.radio("Sentiment",("positive","neutral","negative"))
#somehow u should match random_tweet radio button with our data.query("airline_sentiment"=="positive") etc
#part. to do that u should add "@" (to be able to access var we're comparing against (random_tweet) we need to
#add "@" w var name)
st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iloc[0,0])
#also when we show tweets we dont want certain tweets we want sample/rand tweets so .sample() is added.
#however there is also 1 more thing is needed.we dont want df or series etc.we want txt itself.thus i add iloc


#its time for some plotly stuff.
#but first lets start w a sidebar. also we ll add a dropdown that ll make u choose either hist or piechart.
st.sidebar.markdown("### Number of tweets by sentiment")
select=st.sidebar.selectbox("Visualization type",["Histogram","Pie chart"],key="1")
#key allows us to use sidebar.selectbox widget once again in our code.w "key", streamlit use sidebar.selectbox
#widget once again in our code. (maybe for dif vis for ex, or anything) now slit doesnt confuse w another sbox

#now adding piechart and hist steps (we add a dropdown on prev step)
sentiment_count=data["airline_sentiment"].value_counts()
#st.write(sentiment_count)
#u can see from st.write(sentiment_count) that index has the "neg,pos,neutral" & actual tweet counts are vals.
#we need a TIDY DATAFRAME. one col as "sentiment type" and counts are on the other col. these is what px need
#as an input. tidy format.
sentiment_count=pd.DataFrame({"Sentiment":sentiment_count.index,"Tweets":sentiment_count.values})

#also sts its good idea to hide your visualization to avoid too much clutter. by default checkbox is checked.
#and we say "its True(checked) by default however do these things if its not checked".if hide button nt chckd
if not st.sidebar.checkbox("Hide",True):
    st.markdown("### Number of tweets by sentiment")
    if select == "Histogram":
        fig=px.bar(sentiment_count,x="Sentiment",y="Tweets",color="Tweets",height=500)
        st.plotly_chart(fig)
    else:
        fig=px.pie(sentiment_count,values="Tweets",names="Sentiment")
        st.plotly_chart(fig)


#now here's the new situation, people tweet from where? location map based on tweets.. when & where
#2 conditions are enough for map plotting. longitude and latitude. also ofc there ll be no missing data. if
#these 2 conds are met then st.map(data) will be nough for plotting a map.

#st.map(data)
#if u code this. u see that this data needs  a filtering as time of the days.
st.sidebar.subheader("When and where are users tweeting from?")
hour=st.sidebar.slider("Hour of day",0,23)
#new slider appeared. its a slider. but you can use .number_input() so it becomes counter like. number changes
#with +/- however it should be started from 1 and end w 24. (streamlit malfunction,they ll change this later)
modified_data=data[data["tweet_created"].dt.hour == hour]

#hide map by default.cbox has "Close" txt on it.if u unchecked it u could see the map.
if not st.sidebar.checkbox("Close",True,key="1"):
    st.markdown("### Tweets locations based on the time of day")
    st.markdown("%i tweets between %i:00 and %i:00" %(len(modified_data),hour,(hour+1)%24))
    #last %24 is used as mod24. so we dont show out of bounds value. when u slc 23-24 it ll be 23-00.
    st.map(modified_data)
    #also we can add a checbox that is unchecked by default(false), but when checked,it show the modified_data
    #remember that its unchecked by default(false) so u have to checked it.checkbox returns True if u chckd it
    if st.sidebar.checkbox("Show modified data",False):
        st.write(modified_data)


#ITS time for plot the # of tweets breaking down by each airline. sentiment broken down by each airline.
#its good idea to add multi-choice label bcs users may want to compare airlines.
st.sidebar.subheader("Breakdown airline tweets by sentiment")
choice=st.sidebar.multiselect("Pick airlines", ("US Airways","United","American","Southwest","Delta","Virgin America"),key=0)

#if no one select anything from multiselect then it ll cause a vis error. to avoid that if statement is ok
#also u need to select count (im not sure about that maybe its by default however teacher selects it maybe
#we could choose some other opts like freq/density) also color. labels is optional, u dont have to it just
#changes the name. st.plotly_chart is for visualization on the webapp
if len(choice) > 0:
    choice_data=data[data.airline.isin(choice)]
    fig_choice=px.histogram(choice_data,x="airline",y="airline_sentiment",histfunc="count",
    color="airline_sentiment",facet_col="airline_sentiment",labels={"airline_sentiment":"tweets"},
    height=800,width=600)
    st.plotly_chart(fig_choice)


#its time for final part. WordCloud. which words are spoken. first radio bar for choosing
st.sidebar.header("Word Cloud")
word_sentiment=st.sidebar.radio("Display wordcloud for which sentiment type?",("positive","neutral","negative"))

if not st.sidebar.checkbox("Close",True,key="3"):
    st.header("Word cloud for %s sentiment" %(word_sentiment))
    df=data[data["airline_sentiment"]==word_sentiment]
    words=" ".join(df["text"])
    processed_words=" ".join([word for word in words.split()\
    if "http" not in word and not word.startswith("@") and word !="RT"])
    wordcloud=WordCloud(stopwords=STOPWORDS,background_color="white",height=640,width=800).generate(processed_words)
    plt.imshow(wordcloud)
    #background color should be white. better visuals. its imp to see that wordcloud has .generate()
    #just to be sure we have no xticks and yticks?? (they may not be necessary.)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()

#explaination for words =" ".join part: creating list of words from that df. we ll add space btw words
#explaination for processed_words part: links,http etc shouldnt be used in wordcloud. also common symbols like
# @ etc. words.split and if http not in word and word isnt starting w "@" and word shouldnt include "RT"

#also last st.pyplot() part is for visualization on webapp
