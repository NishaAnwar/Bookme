from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import regex as re


#Embedding Model
model = SentenceTransformer('all-MiniLM-L12-v2')
 
#Now we will create chromadb client
client = chromadb.Client()

#Create or get a collection
collection = client.get_or_create_collection(name="Bookme_info")

text = "BookMe.pk is Pakistan’s leading all‑in‑one online booking platform, " \
"designed to simplify and enhance the travel and entertainment experience of" \
" millions of users across the country. From booking online movie tickets at the " \
"latest cinema releases to securing bus journeys for intercity travel, BookMe.pk " \
"offers a seamless, user‑friendly experience with just a few taps. The app aggregates " \
"options from major bus operators, allowing users to compare schedules, prices, seat types, " \
"and amenities—making travel planning efficient and transparent. Beyond transportation," \
" BookMe.pk also facilitates hotel reservations, event bookings, and even amusement " \
"park entry tickets, enabling users to explore local attractions and events effortlessly. " \
"With integrated features like digital ticket storage, live updates, hassle‑free cancellations," \
" and reliable customer support, it ensures a stress‑free experience. The platform supports " \
"secure payment options including Visa, MasterCard, Mobile Wallets, and cash on delivery. " \
"Its modern interface, real‑time availability tracking, and personalized offers make it a " \
"popular choice among tech‑savvy Pakistanis. Whether it's planning a weekend getaway, watching " \
"the latest blockbuster, attending a concert, or visiting a theme park, BookMe.pk is the go‑to app" \
" for convenient, reliable, and cost‑effective booking across Pakistan. Explore the country with " \
"confidence—download BookMe.pk today and unlock a world of possibilities at your fingertips."

rawSentences = re.split(r'(?<=[.?!])\s+', text.strip())
sentences = [s.lower() for s in rawSentences if s.strip()]


#Generating Embeddings and adding them to chromadb
embeddings = model.encode(sentences)

for i, sentence in enumerate(sentences):

#Adding to collections
    collection.upsert(documents=[sentence],
                   ids=[str(i)],
                   embeddings=[embeddings[i].tolist()]
               )

while(True):
    print("Let's talk with BookMe's Chatbot.....")
    print("1️.Chat")
    print("2️.Exit")
    try:
        userChoice = input("Enter your choice!")
        if userChoice == "1":
            userInput = input("Enter you query!").lower()
            inputVectorEmbedding = model.encode([userInput])
            #Searching in Chromadb for matching vector similar to userInput
            results = collection.query(query_embeddings=inputVectorEmbedding.tolist(), n_results=5)
            if results and results["documents"] and results["documents"][0]:
                bestMatch = results["documents"][0]
                #best_embedding = model.encode([bestMatch])
                #Now we will calculate cosine similarity between userInput and bestMatch
                similarityScore = []
                for match in bestMatch:
                    matchEmbedding=model.encode([match])
                    score=cosine_similarity(inputVectorEmbedding,matchEmbedding)[0][0]
                    similarityScore.append((match,score))
                sortedResults=sorted(similarityScore,key=lambda x:x[1], reverse=True)
                print("\n🔍 Top 5 Matches:")
                for i, (doc, score) in enumerate(sortedResults, 1):
                    print(f"{i}. {doc} (Cosine Similarity: {score:.3f})")
            else:
                print("No relevant information found in the database.")
        elif userChoice == "2":
            break
        else:
            print("Invalid option. Please enter 1 or 2")
    except Exception as e:
        print("Error:", e)
