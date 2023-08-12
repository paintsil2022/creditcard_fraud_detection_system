from typing import Union
from fastapi import FastAPI
from generate_model import train_test_model
app = FastAPI()


@app.get("/", )
def accuracy():

    return train_test_model()