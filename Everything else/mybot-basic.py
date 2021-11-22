#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design
"""
#######################################################
# Initialise Wikipedia agent
#######################################################
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

#######################################################
# Initialise json agent
#######################################################
import json, requests

#######################################################
#  Initialise NPL
#######################################################
import nltk
v = """
Melbourne => {}
Bahrain => {}
Hanoi => {}
Shanghai => {}
Zandvoort => {}
Catalunya => {}
Monaco => {}
Baku => {}
GillesVilleneuve => {}
PaulRicard => {}
RedBull => {}
Silverstone => {}
Hungaroring => {}
Spa => {}
Monza => {}
MarinaBay => {}
Sochi => {}
Suzuka => {}
Americas => {}
HermanosRodriguez => {}
JoseCarlos => {}
YasMarina => {}
Australia => Australia
Bahrain => Bah
Vietnam => Vie
China => Chi
Dutch => Dut
Spain => Spa
Monaco => Mon
Azerbaijan => Aze
Canada => Can
France => Fra
Austria => Austria
Britian => Bri
Hungary => Hun
Belgium => Bel
Italy => Ita
Singapore => Sin
Russia => Rus
Japan => Jap
America => Ame
Mexico => Mex
Brazil => Bra
AbuDhabi => Abu
be_in => {}
"""
folval = nltk.Valuation.fromstring(v)
grammar_file = 'simple-sem.fcfg'
objectCounter = 0

#######################################################
#  Initialise AIML agent
#######################################################
import aiml
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="mybot-basic.xml")

#######################################################
#   Imports for TF-IDF Function
#######################################################
import pandas as pd
import numpy as np
import operator ,os
from sklearn.feature_extraction.text import TfidfVectorizer

#######################################################
#   Imports for loading Convolutional Network model
#######################################################
from keras.models import load_model
from tensorflow.examples.tutorials.mnist import input_data

#######################################################
#   Initialise csv
#######################################################
filepath='F1QA.csv'
csv_reader =pd.read_csv(filepath)
question_list = csv_reader[csv_reader.columns[0]].values.tolist()
answers_list  = csv_reader[csv_reader.columns[1]].values.tolist()


#######################################################
# Welcome user
#######################################################
print("Welcome to the Formula 1 chat bot. Please feel free to ask questions about",
      "Constructors, The scoring System and Past world driver chamipions.",
      "Or if you have a more generic question feel free to ask as I will search wikipedia for the relebent information.")
#######################################################
# Main loop
#######################################################
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print (params[1])
            break
        elif cmd == 1:
            wpage = wiki_wiki.page(params[1])
            if wpage.exists():
                print(wpage.summary)
                print("Learn more at", wpage.canonicalurl)
            else:
                print("Sorry, I don't know what that is.")
        elif cmd == 2:
            succeeded = False
            api_url = r"http://ergast.com/api/f1/constructors/"
            response = requests.get(api_url + params[1] + r".json")
            if response.status_code == 200:
                response_json = json.loads(response.content)
                if response_json:
                    try:
                        nationality = response_json ['MRData']['ConstructorTable']['Constructors'][0]['nationality']
                        name = response_json ['MRData']['ConstructorTable']['Constructors'][0]['name']
                        print(name, "'s nationality is ", nationality)
                    except:
                        print("Sorry, I could not resolve the constructors name you gave me.")
                    succeeded = True
            if not succeeded:
                print("Sorry, I could not resolve the constructors name you gave me.")
        elif cmd == 3:
            model = load_model("model.h5")
            mnist = input_data.read_data_sets('./data')
            x_test, y_test = mnist.test.next_batch(1000)
            model.summary()
            y_pred = model.predict_classes(x_test.reshape((-1,28,28,1)))
            print("\n Accuracy: ",np.count_nonzero(y_pred == y_test)/len(y_test))

        elif cmd == 4: # THE * CIRCUIT IS IN *
            o = 'o' + str(objectCounter)
            objectCounter += 1
            folval['o' + o] = o #insert constant
            if len(folval[params[1]]) == 1: #clean up if necessary
                if ('',) in folval[params[1]]:
                   folval[params[1]].clear()
            folval[params[1]].add((o,)) #insert type of plant information
            if len(folval["be_in"]) == 1: #clean up if necessary
                if ('',) in folval["be_in"]:
                    folval["be_in"].clear()
            folval["be_in"].add((o, folval[params[2]])) #insert location
            
        elif cmd == 5: #IS * CIRCUIT IN *
            g = nltk.Assignment(folval.domain)
            m = nltk.Model(folval.domain, folval)
            sent = 'some ' + params[1] + ' are_in ' + params[2]
            results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
            if results[2] == True:
                print("Yes.")
            else:
                print("No.")
                
        elif cmd == 6: #IS * CIRCUIT ONLY IN *
            g = nltk.Assignment(folval.domain)
            m = nltk.Model(folval.domain, folval)
            sent = 'all ' + params[1] + ' are_in ' + params[2]
            results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
            if results[2] == True:
                print("Yes.")
            else:
                print("No.")
                
        elif cmd == 7: # WHAT CIRCUITS ARE IN *
            g = nltk.Assignment(folval.domain)
            m = nltk.Model(folval.domain, folval)
            e = nltk.Expression.fromstring("be_in(x," + params[1] + ")")
            sat = m.satisfiers(e, "x", g)
            if len(sat) == 0:
                print("None.")
            else:
                #find satisfying objects in the valuation dictionary,
                #and print their type names
                sol = folval.values()
                for so in sat:
                    for k, v in folval.items():
                        if len(v) > 0:
                            vl = list(v)
                            if len(vl[0]) == 1:
                                for i in vl:
                                    if i[0] == so:
                                        print(k)
                                        break
        #FROM HERE 
        elif cmd == 8:
            runGame()
        
        elif cmd == 99:
            query =userInput
            vectorizer = TfidfVectorizer(min_df=0, ngram_range=(2, 4), strip_accents='unicode',norm='l2' , encoding='ISO-8859-1')
            X_train = vectorizer.fit_transform(np.array([''.join(que) for que in question_list]))
            X_query=vectorizer.transform([query])
            XX_similarity=np.dot(X_train.todense(), X_query.transpose().todense())
            XX_sim_scores= np.array(XX_similarity).flatten().tolist()
            dict_sim= dict(enumerate(XX_sim_scores))
            sorted_dict_sim = sorted(dict_sim.items(), key=operator.itemgetter(1), reverse =True)
            if sorted_dict_sim[0][1]==0:
                print("Sorry, I did not get that please ask again")
            elif sorted_dict_sim[0][1]>0:
                print (answers_list [sorted_dict_sim[0][0]])
    else:
        print(answer)


#AND HERE
#REFERENCE https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
def runGame():
    import gym

    env = gym.make("Taxi-v2").env

    env.reset() # reset environment to a new, random state
    env.render()

    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))


    state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
    print("State:", state)

    env.s = state
    env.render()

    q_table = np.zeros([env.observation_space.n, env.action_space.n])


    """Training the agent"""

    import random
    from IPython.display import clear_output

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    for i in range(1, 100001):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
    
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action) 
        
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
        
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1
        
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.\n")


    """Evaluate agent's performance after Q-learning"""

    total_epochs, total_penalties = 0, 0
    episodes = 1000000

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
    
        done = False
    
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
