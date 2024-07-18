from flask import Flask,request,jsonify
from flask_cors import CORS
from recommendation import movie_recommender as recommendation

app=Flask(__name__)
CORS(app)
@app.route('/movie',methods=['GET'])
def recommend_movies():
    movie_title = request.args.get('title')
    if not movie_title:
        return jsonify({'error': 'No movie title provided'}), 400

    try:
        recommendations = recommendation(movie_title)
        return jsonify({'recommendations': recommendations}), 200
    except IndexError:
        return jsonify({'error': 'Movie title not found'}), 404

if __name__=='__main__':
    app.run(port=5000,debug=True)
