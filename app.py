#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import render_template, request
import joblib

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        income= request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        income = float(income)
        age = float(age)
        loan = float(loan)
        print(income, age, loan)
        model1 = joblib.load("CCD_Reg")
        pred1 = model1.predict ([[income, age, loan]])
        s1 = "Prediction based on regression is " + str(pred1)
        
        model2 = joblib.load("CCD_DT")
        pred2 = model1.predict ([[income, age, loan]])
        s2 = "Prediction based on decision tree is " + str(pred2)
       
    
        model3 = joblib.load("CCD_RF")
        pred3 = model3.predict ([[income, age, loan]])
        s3 = "Prediction based on random forest is " + str(pred3)
        
        model4 = joblib.load("CCD_NN")
        pred4 = model4.predict ([[income, age, loan]])
        s4 = "Prediction based on neural network is " + str(pred4)
        
        model5 = joblib.load("CCD_XGB")
        pred5 = model5.predict ([[income, age, loan]])
        s5 = "Prediction based on gradient boosting is " + str(pred5)
        return(render_template("index.html", result1=s1, result2=s2, result3=s3, result4=s4, result5=s5))
    else:
        return(render_template("index.html", result1="2", result2="2", result3="2", result4="2", result5="2"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




