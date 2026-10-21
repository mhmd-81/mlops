from fastapi import FastAPI

app = FastAPI(title='iris_model')

@app.get('/health')
def health_check():
    return{"status":"healthy"}

@app.get('/')
def home():
    return{'message':'welcome to iris app'}

@app.post('/hello')
def say_hello(name:str,age:int):
    return {"message":f"hello {name} who is {age} years old"}