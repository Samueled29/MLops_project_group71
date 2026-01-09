from fruit_and_vegetable_disease.model import Model
from fruit_and_vegetable_disease.data import MyDataset

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
