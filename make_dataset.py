"""
Generates data/movies.csv — the 50-movie curated sample dataset used to
validate the full pipeline end-to-end.

Run:
    python3 make_dataset.py

The CSV schema (title, genres, overview) matches MovieLens / TMDB, so
the same recommender.py code works against either source with no changes.
"""

import os
import pandas as pd


def create_sample_dataset():
    data = [
        # Action | Sci-Fi | Thriller
        {"title": "Inception", "genres": "Action|Sci-Fi|Thriller",
         "overview": "A thief who steals secrets through dream-sharing technology is given the task of planting an idea into the mind of a CEO."},
        {"title": "The Matrix", "genres": "Action|Sci-Fi",
         "overview": "A computer programmer discovers reality is a simulation and joins a rebellion against the machines controlling it."},
        {"title": "Gravity", "genres": "Drama|Sci-Fi|Thriller",
         "overview": "Two astronauts work together to survive after an accident leaves them adrift in space."},
        {"title": "Interstellar", "genres": "Adventure|Drama|Sci-Fi",
         "overview": "A team of explorers travels through a wormhole in space in an attempt to ensure humanity's survival."},
        {"title": "Blade Runner 2049", "genres": "Drama|Mystery|Sci-Fi",
         "overview": "A young blade runner discovers a secret that leads him to track down former blade runner Rick Deckard."},
        {"title": "Mad Max: Fury Road", "genres": "Action|Adventure|Sci-Fi",
         "overview": "In a post-apocalyptic wasteland, Max teams with a mysterious woman to flee a tyrannical warlord."},
        {"title": "Iron Man", "genres": "Action|Adventure|Sci-Fi",
         "overview": "After being held captive, billionaire engineer Tony Stark creates a unique weaponized suit of armor."},
        {"title": "Black Panther", "genres": "Action|Adventure|Sci-Fi",
         "overview": "T'Challa returns home to Wakanda to take his place as king, but his right to rule is challenged."},
        {"title": "Avengers: Endgame", "genres": "Action|Adventure|Sci-Fi",
         "overview": "The Avengers assemble once more to reverse the devastating events of Thanos with the help of remaining allies."},

        # Action | Crime | Drama | Thriller
        {"title": "The Dark Knight", "genres": "Action|Crime|Drama",
         "overview": "Batman faces the Joker, a criminal mastermind who plunges Gotham into chaos and anarchy."},
        {"title": "Sicario", "genres": "Action|Crime|Drama",
         "overview": "An idealistic FBI agent is enlisted by a government task force to aid in the war against drugs."},
        {"title": "John Wick", "genres": "Action|Crime|Thriller",
         "overview": "An ex-hitman comes out of retirement to track down the gangsters who killed his dog and stole his car."},
        {"title": "No Country for Old Men", "genres": "Crime|Drama|Thriller",
         "overview": "Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong and finds two million dollars."},
        {"title": "Pulp Fiction", "genres": "Crime|Drama",
         "overview": "The lives of two mob hitmen, a boxer, and a gangster's wife intertwine in four tales of violence and redemption."},
        {"title": "Skyfall", "genres": "Action|Adventure|Thriller",
         "overview": "James Bond's loyalty to M is tested as her past comes back to haunt her in the form of a dangerous new villain."},
        {"title": "The Bourne Identity", "genres": "Action|Mystery|Thriller",
         "overview": "A man is rescued at sea with no memory and two bullets in his back, possessing dangerous combat skills."},
        {"title": "Gladiator", "genres": "Action|Adventure|Drama",
         "overview": "A Roman general is betrayed and his family murdered, then rises as a gladiator to exact his revenge."},

        # Animation | Adventure | Comedy | Family
        {"title": "Toy Story", "genres": "Animation|Adventure|Comedy|Family",
         "overview": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room."},
        {"title": "Finding Nemo", "genres": "Animation|Adventure|Comedy|Family",
         "overview": "After his son is captured in the Great Barrier Reef, a timid clownfish sets out on a journey to bring him home."},
        {"title": "Up", "genres": "Animation|Adventure|Comedy|Family",
         "overview": "78-year-old Carl Fredricksen travels to Paradise Falls in his house equipped with balloons, inadvertently taking a young stowaway."},
        {"title": "Zootopia", "genres": "Animation|Adventure|Comedy|Family",
         "overview": "In a city of anthropomorphic animals, a rookie bunny cop and a cynical con artist fox must work together to uncover a conspiracy."},
        {"title": "Inside Out", "genres": "Animation|Adventure|Comedy|Family",
         "overview": "After young Riley is uprooted from her Midwest life and moved to San Francisco, her emotions conflict on how best to navigate a new city."},
        {"title": "Shrek", "genres": "Animation|Adventure|Comedy|Family",
         "overview": "A mean lord exiles fairytale creatures to the swamp of a grumpy ogre, who must go on a quest to rescue a princess."},
        {"title": "Frozen", "genres": "Animation|Adventure|Comedy|Family|Musical",
         "overview": "When the queen of Arendelle inadvertently plunges her land into eternal winter, her sister embarks on a journey to break the spell."},
        {"title": "Coco", "genres": "Animation|Adventure|Comedy|Family|Musical",
         "overview": "Aspiring musician Miguel is transported to the Land of the Dead and seeks the blessing of his deceased great-great-grandfather."},

        # Horror | Mystery | Thriller
        {"title": "Get Out", "genres": "Horror|Mystery|Thriller",
         "overview": "A young African-American visits his white girlfriend's parents for the weekend, where his simmering uneasiness reaches a boiling point."},
        {"title": "The Conjuring", "genres": "Horror|Mystery|Thriller",
         "overview": "Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence in their farmhouse."},
        {"title": "It", "genres": "Horror|Thriller",
         "overview": "In 1989, a group of bullied kids band together to destroy a shape-shifting monster disguised as a clown that preys on children."},
        {"title": "A Quiet Place", "genres": "Horror|Sci-Fi|Thriller",
         "overview": "A family is forced to live in near-silence while hiding from creatures that hunt exclusively by sound."},
        {"title": "Hereditary", "genres": "Drama|Horror|Mystery",
         "overview": "When the matriarch of the Graham family passes away, her daughter and grandchildren begin to unravel cryptic and terrifying secrets."},
        {"title": "Midsommar", "genres": "Drama|Horror|Mystery",
         "overview": "A couple travel to Sweden to visit a rural folk festival that takes a dark turn when the locals reveal their true traditions."},
        {"title": "Us", "genres": "Horror|Mystery|Thriller",
         "overview": "A family's vacation turns violent and surreal when mysterious doppelgangers begin to terrorize them."},

        # Drama | Romance
        {"title": "La La Land", "genres": "Comedy|Drama|Musical|Romance",
         "overview": "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their professional dreams."},
        {"title": "Titanic", "genres": "Drama|Romance",
         "overview": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the ill-fated R.M.S. Titanic."},
        {"title": "Forrest Gump", "genres": "Drama|Romance",
         "overview": "The presidencies of Kennedy and Johnson, the Vietnam War, and other events unfold through the perspective of an Alabama man with an extraordinary life."},
        {"title": "Eternal Sunshine of the Spotless Mind", "genres": "Drama|Romance|Sci-Fi",
         "overview": "When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories."},
        {"title": "Her", "genres": "Drama|Romance|Sci-Fi",
         "overview": "In a near future, a lonely writer develops an unlikely relationship with an operating system designed to meet his every need."},
        {"title": "The Notebook", "genres": "Drama|Romance",
         "overview": "An elderly man reads from a notebook to a woman diagnosed with dementia, telling a story of two young lovers."},

        # Drama | Crime | Thriller
        {"title": "Fight Club", "genres": "Drama|Thriller",
         "overview": "An insomniac office worker forms an underground fight club with a soap salesman whose philosophy turns increasingly dangerous."},
        {"title": "The Shawshank Redemption", "genres": "Drama",
         "overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."},
        {"title": "The Silence of the Lambs", "genres": "Crime|Drama|Thriller",
         "overview": "A young FBI cadet must confide in an incarcerated cannibalistic killer to catch a serial murderer who skins his victims."},
        {"title": "Gone Girl", "genres": "Drama|Mystery|Thriller",
         "overview": "With his wife's disappearance having become the focus of an intense media circus, a man sees the spotlight turned on him."},

        # Adventure | Drama
        {"title": "The Lord of the Rings: The Fellowship of the Ring", "genres": "Action|Adventure|Drama|Fantasy",
         "overview": "A meek hobbit and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth."},
        {"title": "Jurassic Park", "genres": "Action|Adventure|Sci-Fi|Thriller",
         "overview": "A pragmatic paleontologist tours an almost complete theme park of cloned dinosaurs that escapes containment."},
        {"title": "The Avengers", "genres": "Action|Adventure|Sci-Fi",
         "overview": "Earth's mightiest heroes must come together to stop Loki and his alien army from enslaving humanity."},

        # Comedy
        {"title": "The Grand Budapest Hotel", "genres": "Adventure|Comedy|Crime",
         "overview": "The adventures of Gustave H, a legendary concierge at a famous European hotel, and Zero Moustafa, his lobby boy."},
        {"title": "Knives Out", "genres": "Comedy|Crime|Drama|Mystery",
         "overview": "A detective investigates the death of a patriarch of an eccentric, combative family when a renowned crime novelist is found dead."},
        {"title": "Parasite", "genres": "Comedy|Drama|Thriller",
         "overview": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan."},
        {"title": "Jojo Rabbit", "genres": "Comedy|Drama|War",
         "overview": "A lonely German boy's world view is turned upside down when he discovers his single mother is hiding a Jewish girl in their attic."},
    ]

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv("data/movies.csv", index=False)
    print(f"Sample dataset created at data/movies.csv ({len(df)} movies, "
          f"{df['genres'].str.split('|').explode().nunique()} unique genres)")


if __name__ == "__main__":
    create_sample_dataset()
