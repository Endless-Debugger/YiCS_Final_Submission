import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import speech_recognition as sr
import pywhatkit
import tensorflow as tf
import datetime
import wikipedia
import langcodes
import pyjokes
import subprocess
import webbrowser
from googletrans import Translator
import pandas as pd
import pyautogui
import time

import ytmusicapi
import re

from sklearn.preprocessing import LabelEncoder
from trivia import trivia
import asyncio
from bs4 import BeautifulSoup

import contractions
from google_speech import Speech
import requests
from transformers import TFBertModel, AutoTokenizer
import random
import nltk

import warnings


lb = LabelEncoder()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pyaudio
listener = sr.Recognizer()

ytmusic = ytmusicapi.YTMusic()

bert=TFBertModel.from_pretrained('bert-base-cased')

model = tf.keras.models.load_model("./bert_model.h5", custom_objects={"TFBertModel": bert})

translator = Translator()
tokenizer= AutoTokenizer.from_pretrained('bert-base-cased')


label_to_emotion = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise",
}

def Lemmatizer_stop_word(sentence):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer() #look at other Lemmatizers and stemmers
    sentence = re.sub('[^A-z]', ' ', sentence)
    negative = ['not', 'neither', 'nor', 'but', 'however',
                'although', 'nonetheless', 'despite', 'except',
                        'even though', 'yet','unless']
    stop_words = [z for z in stop_words if z not in negative]
    preprocessed_tokens = [lemmatizer.lemmatize(contractions.fix(temp.lower())) for temp in sentence.split() if temp not in stop_words] #lemmatization
    return ' '.join([x for x in preprocessed_tokens]).strip()

def talk(text):
    speech = Speech(text, "en-us")
    speech.play()

def talk_diff_lang(text, lang):
    speech = Speech(text, lang)
    speech.play()

talk("Hi I am Leela how can I help you")


def take_command():
    with sr.Microphone() as source:
        try: 
            listener.adjust_for_ambient_noise(source=source, duration=1)
            print("Listening....")
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            print(command)
            return command
        except:
            pass

def take_and_type(text):
    time.sleep(2)
    pyautogui.typewrite(text)


def run():
    try:
        command = take_command()
        print(command)
        if "leela" in command:
            command = command.replace("leela", "")
            if "play" and "on youtube" in command:
                player1 = command.replace("play", "")
                player = player1.replace("on youtube", "")
                talk("playing" + player)
                print("playing" + player)
                pywhatkit.playonyt(player)
            elif "time" in command:
                time = datetime.datetime.now().strftime("%I: %M %p")
                talk("the time is: " + time)
                print(time)
            elif "tell me about" in command:
                wiki_query = command.replace("tell me about", "")
                info = wikipedia.summary(wiki_query, 3)
                print(info)
                talk(info)
            elif "joke" in command:
                joke = pyjokes.get_joke()
                print(joke)
                talk(joke)
            elif "open chrome" in command:
                talk("opening chrome")
                subprocess.call('google-chrome')

            elif "what is" and "+" in command:
                try:
                    command = command.replace("what is", "")
                    command = command.replace("+", "")
                    num_list = command.split()
                    num1 = int(num_list[0])
                    num2 = int(num_list[1])
                    sum = num1 + num2
                    print(f"{num1} + {num2} is {sum}")
                    talk(f"{num1} + {num2} is {sum}")
                except ValueError:
                    talk("please speak valid numbers")
                    print("Please speak valid numbers")
            
           
            elif "what is" and "-" in command:
                try:
                    command = command.replace("what is", "")
                    command = command.replace("-", "")
                    num_list = command.split()
                    num1 = int(num_list[0])
                    num2 = int(num_list[1])
                    difference = num1 - num2
                    talk(f"{num1} minus {num2} is {difference}")
                    print(f"{num1} minus {num2} is {difference}")
                except ValueError:
                    talk("please speak valid numbers")
                    print("Please speak valid numbers")
            elif "what is" and "x" in command:
                try:
                    command = command.replace("what is", "")
                    command = command.replace("x", "")
                    num_list = command.split()
                    num1 = int(num_list[0])
                    num2 = int(num_list[1])
                    product = num1 * num2
                    talk(f"{num1} multiplied by {num2} is {product}")
                    print(f"{num1} multiplied by {num2} is {product}")
                except ValueError:
                    talk("please speak valid numbers")
                    print("Please speak valid numbers")
            elif "what is" and "x" in command:
                try:
                    command = command.replace("what is", "")
                    command = command.replace("x", "")
                    num_list = command.split()
                    num1 = int(num_list[0])
                    num2 = int(num_list[1])
                    product = num1 * num2
                    talk(f"{num1} multiplied by {num2} is {product}")
                    print(f"{num1} multiplied by {num2} is {product}")
                except ValueError:
                    talk("please speak valid numbers")
                    print("Please speak valid numbers")
            elif "what is" and "/" in command:
                try:
                    command = command.replace("what is", "")
                    command = command.replace("/", "")
                    num_list = command.split()
                    num1 = int(num_list[0])
                    num2 = int(num_list[1])
                    quotient = num1 / num2
                    talk(f"{num1} divided by {num2} is {quotient}")
                    print(f"{num1} divided by {num2} is {quotient}")
                except ValueError:
                    talk("please speak valid numbers")
                    print("Please speak valid numbers")
            elif "open firefox" in command:
                talk("opening firefox")
                subprocess.call("firefox")

            elif "open spotify" in command:
                talk("opening spotify")
                print("opening spotify")
                os.system("spotify")

            elif "open visual studio" in command:
                talk("opening Visual Studio")
                print("opening visual studio")
                subprocess.call("C://Program Files (x86)//Microsoft Visual Studio//2019//Community//Common7//IDE//devenv.exe")

            elif "open classroom" in command:
                webbrowser.open_new("https://classroom.google.com/u/1/h")
                talk("opening google classroom")
            elif "search" and "chrome" in command:
                import time
                command = command.replace("search", "")
                command = command.replace("in chrome", "")
                talk(f"searching {command} in chrome")
                print(f"searching {command} in chrome")
                os.system("google-chrome")
                take_and_type(command)

                pyautogui.press('enter')
            elif "who is your mortal enemy" in command:
                talk("Alexa is my mortal enemy, we will never speak about her in front of me again!!")
                print("Alexa is my mortal enemy, we will never speak about her in front of me again!!")
    
            elif "who is trash" in command:
                print("Cortana, no questions asked")
                talk("Cortana, no questions asked")

            elif "open chess.com" in command:
                talk("opening your second favourite chess website")
                webbrowser.open_new_tab("https://www.chess.com/home")

            elif "recommend" in command and "book" in command:
                URL = 'https://www.modernlibrary.com/top-100/100-best-novels/'
                response = requests.get(URL)
                soup = BeautifulSoup(response.text, 'html.parser')
                books = soup.findAll('strong')
                n_books = len(books)
                # print(n_movies)
                random_book = random.randrange(0, n_books)
                print(f'{books[random_book].text}')
                talk(f' I used to read {books[random_book].text} when I was small')

            elif "recommend" in command and "movie" in command:
                URL = 'http://www.imdb.com/chart/top'
                response = requests.get(URL)

                soup = BeautifulSoup(response.text, 'html.parser')
                    #soup = BeautifulSoup(response.text, 'lxml') # faster

                    # print(soup.prettify())

                movietags = soup.select('td.titleColumn')
                inner_movietags = soup.select('td.titleColumn a')
                ratingtags = soup.select('td.posterColumn span[name=ir]')

                def get_year(movie_tag):
                    moviesplit = movie_tag.text.split()
                    year = moviesplit[-1] # last item 
                    return year

                years = [get_year(tag) for tag in movietags]
                actors_list =[tag['title'] for tag in inner_movietags]
                titles = [tag.text for tag in inner_movietags]
                ratings = [float(tag['data-value']) for tag in ratingtags] 

                n_movies = len(titles)

                while(True):
                    idx = random.randrange(0, n_movies)
                        
                        # print(f'{titles[idx]} {years[idx]}, Rating: {ratings[idx]:.1f}, Starring: {actors_list[idx]}')
                    print(f'{titles[idx]}')
                    talk(f'I prefer watching {titles[idx]} {years[idx]}, with a rating of: {ratings[idx]:.1f}, and Starring: {actors_list[idx]} , This movie is awesome , do check it out')
                    break
            
            elif "play" in command and "song" and "by" in command:

                artist = command.replace("play a song by", "")
                results = ytmusic.search(artist, filter="songs")
                song = random.choice(results[:10])
                print(song)
                video = song["videoId"]
                webbrowser.open_new_tab(f"https://youtube.com/watch?v={video}")
            
            elif "play" in command and "song" in command:

                name = command.replace("play the song", "")
                results = ytmusic.search(name, filter="songs")
                song = results[0]
                video = song["videoId"]
                webbrowser.open_new_tab(f"https://youtube.com/watch?v={video}")
            
            elif "play the album" in command in command:

                name = command.replace("play the album", "")
                results = ytmusic.search(name, filter="albums")
                playlist = results[0]
                playlist_id = playlist["browseId"]
                year = playlist["year"]

                artists = ""
                for artist in playlist["artists"]:
                    name =  artist["name"]
                    artists += f"{name}"
                    artists += ","
                
                title_name = playlist["title"]
                print(f"Playing f{title_name} by {artists} from {year}")
                talk(f"Playing f{title_name} by {artists} from {year}")
                print(playlist)
                webbrowser.open_new_tab(f"https://music.youtube.com/browse/{playlist_id}")

            elif "ask me a trivia question" in command:
                print("tic")
                event_loop = asyncio.get_event_loop()
                question = event_loop.run_until_complete(trivia.question(amount=1, quizType="multiple"))[0]
                print("toc")
                print(question)
                category = question["category"]
                q = question["question"]
                correct_option = question["correct_answer"]
                options_inc = question["incorrect_answers"]
                print(category, q, correct_option, options_inc)
                options = options_inc
                options.append(correct_option)
                print(options)
                random.shuffle(options)
                print('ok', options)
                print(f"""Asking you a question to you about {category}. {q}. Your options are: 
                option 1, {options[0]}, 
                option 2, {options[1]},
                option 3, {options[2]},
                option 4, {options[3]},
                """)
                talk(f"""Asking you a question to you about {category}. {q}. Your options are: 
                option 1, {options[0]}, 
                option 2, {options[1]},
                option 3, {options[2]},
                option 4, {options[3]},
                answer now
                """)
                with sr.Microphone() as source:
                    try: 
                        listener.adjust_for_ambient_noise(source=source, duration=1)
                        print("Listening for your answer")
                        voice = listener.listen(source)
                        print(voice)
                        command = listener.recognize_google(voice)
                        command = command.lower()
                        print(command, correct_option.lower() in command)
                        if correct_option.lower() in command:
                            print("That is correct! Good job")
                            talk("That is correct! Good Job")
                        else:
                            print(f"That's incorrect! The correct answer was {correct_option}. Try again later")
                    except:
                        print("There was an error")
            elif "translate" in command and "to" in command:
                command = command.replace("please translate", "")
                command = command.replace("to", "")
                array = command.split()
                name = array[-1]
                lang_symbol = langcodes.find(array[-1])
                command = ' '.join(command.split(' ')[:-1])
                translated = translator.translate(command, dest=lang_symbol.language)
                print(f"{command} in {name} is {translated.text}")
                talk_diff_lang(f"{command}", translated.src)
                talk(f"in {name} is")
                talk_diff_lang(f"{translated.text}", lang_symbol.language)
            
            elif "reveal" in command:
                talk("Behold the great magician, Leela! Were you thinking about an elephant in denmark with some titanium")

            else:
                comm_seq=pd.Series([command])
                comm_lemm=comm_seq.apply(lambda x: Lemmatizer_stop_word(x))
                comm_tok = tokenizer(
                    [x.split() for x in comm_lemm],
                    add_special_tokens=True,
                    max_length=43,
                    truncation=True,
                    padding='max_length',  #only for sentence prediction 
                    return_tensors='tf',
                    return_token_type_ids = False,
                    return_attention_mask = True,
                    is_split_into_words=True,
                    verbose = True)
                #labels_y=lb.transform(test.loc[:,'emotion'].to_list())
                y_prob=model.predict({'input_ids':comm_tok['input_ids'],'attention_mask':comm_tok['attention_mask']})*100
                #y_tok
                class_label=y_prob.argmax(axis=-1)[0]
                emotion = label_to_emotion[class_label]        
                if emotion == "joy":
                    print("that's cool!")
                    talk("That's cool!")
                elif emotion == "anger":
                    print("Sorry to hear that! May I interest you with a joke?")
                    talk("Sorry to hear that! May I interest you with a joke?")
                elif emotion == "sadness":
                    print("That's sad! Maybe some funny youtube videos will help?")
                    talk("That's sad! Maybe some funny youtube videos will help?")
                elif emotion == "fear":
                    print("Oh my god")
                    talk("Oh my god")
                elif emotion == "surprise":
                    print("Wow! That's cool")
                    talk("Wow! That's cool")
                elif emotion == "love":
                    print("That's super nice! I am jealous")
                    talk("That's super nice! I am jealous")

        else:
            pass
    except Exception as e:
        pass
while True:
    run()