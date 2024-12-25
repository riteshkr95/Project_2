from flask import Flask , render_template , request
import joblib

model=joblib.load("log_reg.pkl")

app=Flask(__name__)
@app.route("/")

def index():
    return render_template("index.html")

@app.route("/submit" ,methods=["post"])

def sumbit():
    a=eval(request.form.get("%Red Pixel"))
    b=eval(request.form.get("%Green pixel"))
    c=eval(request.form.get("%Blue pixel"))
    d=eval(request.form.get("Hb"))
    prediction=model.predict([[a,b,b,c]])
    if prediction[0]==0:
        return "you are safe"
    else:
        return "you are not safe"
app.run(debug=True ,port=8989 , host="0.0.0.0")
