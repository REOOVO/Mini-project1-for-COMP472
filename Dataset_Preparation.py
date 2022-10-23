from gzip import open as gopen
import json
import matplotlib.pyplot as plt

DATA_PATH = "goemotions.json.gz"
GRAPH_SAVE_PATH = "graph"


def load_dataset(path):
    data = []
    with gopen(path, 'rb') as f:
        file_content = f.read()
        data = json.loads(file_content)
    return data


def create_graph(datas, save_path):
    emotion, sentiment = datas

    plt.pie(emotion.values(), labels=emotion.keys())
    plt.title('Emotion')
    plt.xlabel('Emotion Type')
    plt.savefig(save_path+'\\emotion.jpg')
    plt.show()

    plt.pie(sentiment.values(), labels=sentiment.keys())
    plt.title('Sentiment')
    plt.xlabel('Sentiment Type')
    plt.savefig(save_path+'\\sentiment.jpg')
    plt.show()


def count_types(data):
    emotion = dict()
    sentiment = dict()
    for item in data:
        emotion[item[1]] = emotion.get(item[1], 0)+1
        sentiment[item[2]] = sentiment.get(item[2], 0)+1
    return emotion, sentiment


def main():
    data = load_dataset(DATA_PATH)
    create_graph(count_types(data), GRAPH_SAVE_PATH)


if __name__ == "__main__":
    main()
