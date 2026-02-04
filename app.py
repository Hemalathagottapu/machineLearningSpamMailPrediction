from flask import Flask,render_template,request
import pickle
tokenizer=pickle.load(open(r"C://Users//Hemalatha_2006//Documents//abcd//hemalatha//MovieReccommendedSystem//flask//models//cv.pkl","rb"))
model=pickle.load(open(r"C:\Users\Hemalatha_2006\Documents\abcd\hemalatha\MovieReccommendedSystem\flask\models\clf.pkl","rb"))
# print("CV FILE LOADED FROM:", tokenizer)
# print("Is fitted?:", hasattr(tokenizer, "vocabulary_"))
app=Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")
@app.route("/predict",methods=['POST'])
def predict():
    email_text=[request.form.get("email-content")]
    tokenize_email=tokenizer.transform(email_text)
    predictions=model.predict(tokenize_email)[0]
    predictions=1 if predictions==1 else -1
    return render_template("index.html",predictions=predictions,email_text=email_text)
if __name__=="__main__":
    app.run(debug=True)
