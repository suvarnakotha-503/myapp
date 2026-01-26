from flask import Flask
app=Flask(__name__)
@app.route("/")
def home():
   return "hello fron ci/cd pipeline"
app.run(host="0.0.0.0", port=5000)

