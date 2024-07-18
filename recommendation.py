import pandas as pd
import re
import numpy as np
import contractions
from sklearn import preprocessing
df=pd.read_csv('D:/python/movie/movies_metadata.csv')
# print(df.head())
import nltk
nltk.download('punkt')
nltk.download('stopwords')
df=df.drop(['adult','belongs_to_collection','imdb_id','poster_path'],axis=1)
df=df.dropna()
df.reset_index(inplace=True, drop=True)
le = preprocessing.LabelEncoder()
f = df[['title', 'tagline', 'overview', 'popularity']]
df.tagline.fillna('', inplace=True)
df['description'] = df['tagline'].map(str) + ' ' + df['overview']
df.dropna(inplace=True)
df['popularity'] = le.fit_transform(df['popularity'].astype(str))
df = df.sort_values(by=['popularity'], ascending=False)
# df.info()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    doc = contractions.fix(doc)
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    #filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

norm_corpus = normalize_corpus(list(df['description']))
# print(len(norm_corpus))
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
tfidf_matrix = tf.fit_transform(norm_corpus)
# tfidf_matrix.shape
from sklearn.metrics.pairwise import cosine_similarity

doc_sim = cosine_similarity(tfidf_matrix)
doc_sim_df = pd.DataFrame(doc_sim)
print(doc_sim_df.head())
movies_list = df['title'].values
# movies_list, movies_list.shape
movie_idx = np.where(movies_list == 'Blade Runner')[0][0]
movie_similarities = doc_sim_df.iloc[movie_idx].values
similar_movie_idxs = np.argsort(-movie_similarities)[1:6]
similar_movies = movies_list[similar_movie_idxs]
def movie_recommender(movie_title, movies=movies_list, doc_sims=doc_sim_df):
    movie_idx = np.where(movies == movie_title)[0][0]
    movie_similarities = doc_sims.iloc[movie_idx].values
    similar_movie_idxs = np.argsort(-movie_similarities)[1:6]
    similar_movies = movies[similar_movie_idxs]
    return similar_movies.tolist()
# print(movies_list)
popular_movies = ['Blade Runner', 'The Haunting of Molly Hartley', 'The Lords of Salem', 'Shaun the Sheep Movie',
              'Choppertown: The Sinners', 'In Custody', 'Everything Must Go', 
              'Young Dragons: Kung Fu Kids II', 'The Great Gatsby', 'The Martian', 'Batman v Superman: Dawn of Justice','Leaves of Grass',
       'Beowulf', 'All the Boys Love Mandy Lane',
       'Christopher and His Kind', 'Attack the Block', 'The Fog',
       'Admiral', 'Baraka', 'Flushed Away', 'Elles', 'Little Birds',
       'The House Bunny', 'Rambo', 'Neighbors 2: Sorority Rising',
       'Alvin and the Chipmunks: The Squeakquel','Funny People', 'Eastern Promises', 'In Bruges', 'Notting Hill',
       'A Dangerous Method', "A Dog's Purpose",
       'The Girl with the Dragon Tattoo', 'Horrible Bosses', 'Eragon',
       'The Duchess', 'The Mechanic', 'The Reaping',
       'Wall Street: Money Never Sleeps', 'Lymelife', 'All Eyez on Me',
       'Knowing', 'Detroit', 'Source Code', 'Crash',
       'The Spy Who Loved Me', 'Live and Let Die', 'Iron Jawed Angels',
       'In a World...', 'Project Nim', 'For Your Eyes Only', 'Starman',
       '30 Minutes or Less', 'Post Grad', 'Little Fockers', 'Gamer','Marley & Me', 'The Colony', 'Nine Dead', 'Boogeyman', 'Air',
       'Arthur', 'Waiting for Forever', 'No Reservations',
       'Happythankyoumoreplease', 'Solomon Kane', 'Fright Night',
       'Animal Kingdom', 'Zero Dark Thirty', 'Time Lapse',
       'American Gangster', 'Case 39', 'Seven Psychopaths',
       'Sky Captain and the World of Tomorrow', 'Severance',
       'The Game Plan', 'According to Greta', 'End of Days', 'Hard Candy',
       'Constantine', 'The Water Diviner', 'Grabbers', 'Pom Poko',
       'In the Valley of Elah', 'Prayers for Bobby', 'Riddick',
       'The Mists of Avalon', 'Nymphomaniac: Vol. II',
       'The Next Three Days', '9th Company', 'The Captains',
       'Into the Woods', 'Haywire', "Grumpy Cat's Worst Christmas Ever",
       'The Promise', 'The Last Airbender', '13 Assassins', 'Rent',
       'J. Edgar', 'Justice League: The Flashpoint Paradox',
       'The Cider House Rules', 'Old Dogs', 'Bronson',
       'The Edge of Seventeen', 'Blitz', 'The Scorpion King','Keeping Up with the Joneses', 'Jonah Hex', 'The Rocker',
       'Journey 2: The Mysterious Island', "Fool's Gold",
       'Million Dollar Arm', 'Victor Frankenstein', 'Heavenly Creatures',
       'March of the Penguins', 'I Love You, Beth Cooper',
       "While We're Young", 'The Walk',
       'Ghost Rider: Spirit of Vengeance', 'Ride Along 2',
       'The Adjustment Bureau', 'I, Frankenstein', 'Grown Ups 2',
       'Tales from Earthsea', 'The Hurt Locker', 'The Way Back',
       'I Am Number Four', 'The Host', 'Hoodwinked Too! Hood VS. Evil',
       'Leatherheads', 'R.I.P.D.', 'Project Almanac', 'Bad Moms',
       "On Her Majesty's Secret Service", "I'm Not There.",
       'In the Land of Women', 'When Marnie Was There', 'Expelled',
       'The Wind Rises', 'The Librarian: The Curse of the Judas Chalice',
       'The Man from Earth', 'The Wolfman', 'Transporter 3', 'Wanderlust',
       'Robot & Frank', 'All About Steve', 'Burlesque', 'Babylon A.D.',
       'The Sum of All Fears', 'Just Go with It',
       'Hunt for the Wilderpeople', 'Reach Me', 'The New Guy',
       'Café Society', 'Gnomeo & Juliet',
       'The Best Exotic Marigold Hotel', 'Pain & Gain',
       'The End of the Tour', 'American Grindhouse', '1408',
       'Swiss Army Man', 'The Fog', 'Dark Places', 'Detour',
       "A Dog's Breakfast", 'Sex and the City', 'Ninja Assassin',
       'G-Force', 'Across the Universe', 'Survival of the Dead',
       'Cop Land', 'Evan Almighty', 'John Dies at the End', 'Moana',
       'Happy Feet Two', 'Hall Pass', 'Ratchet & Clank', 'Step Up 3D',
       'Abraham Lincoln vs. Zombies', 'Invictus', 'Sex Drive',
       'A Hologram for the King', 'Spaceballs', 'Sunshine Cleaning',
       'Duplicity', 'Felon', 'Magnolia', "It's Complicated", 'The Void',
       'Shall We Dance?', 'Head in the Clouds', 'The Bucket List',
       'Tinker Bell', 'After.Life', 'Knocked Up', 'Tracers', 'Religulous',
       'Insidious: Chapter 2', 'The Dukes of Hazzard',
       'The Man from Nowhere', 'Ronal the Barbarian', 'Faster',
       'Two for the Money', 'Brick', 'The Avengers', 'The Circle']

# for movie in popular_movies:
#     print('Movie:', movie)
#     print('Top 5 recommended Movies:', movie_recommender(movie_title=movie, movies=movies_list, doc_sims=doc_sim_df))
#     print()