# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:09:27 2021

@author: ASUS
"""

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import re
import pickle
import lastmodel

st.sidebar.image("Team_Logo1.JPG" , width=300 , caption='Team TRIO')

menu = ["Home" , "Data Visualization" ,"About Us"]

choice = st.sidebar.selectbox("Menu" , menu)

if choice == "Home": #Home
    
    st.markdown("<h1 style='text-align: center; color:#1DA1F2'><b>Twitter Sentiment Analysis Tool For <br> Racism And Sexism</b></h1>", unsafe_allow_html=True)
    
    st.text("")
    
    st.image("Body.jpg" , width = 700)
    
    st.text("")
    
    user_review = st.text_input('Enter The Comment You Want To Test')
    
    if user_review == False:
        st.waring('Please Enter A Comment.')
      
    if st.button("Analyze"):
        prediction = lastmodel.testing(user_review)
        if prediction == 0:
            st.success('Your comment is a positive comment. Racism and Sexism should not be in the world.You are great!')
        elif prediction == 1:
            st.error('Your comment is a negative comment. The day you stop blaming others is the day you begin to discover who you truely are.')

   
elif choice == "About Us":
    
    st.markdown("<h1 style='text-align: center; color: #1DA1F2; font-size: 200%'>Meet The Team</h1>", unsafe_allow_html=True)
    
    st.text("")
    
    col1 , col2 , col3 = st.columns(3)
    col1.image('hhm.png', width = 200)
    col1.write("<p style = 'text-align: left; font-size:110%; color:#1DA1F2'></p>", unsafe_allow_html = True)
    
    col2.image('nmk.png', width = 200)
    col2.write("<p style = 'text-align: left; font-size:110%; color:#1DA1F2'>I am Nyi Min Khant. I am a student from UTYCC. I made the User Interface of this software.</p>", unsafe_allow_html = True)
    
    col3.image('tyn.png', width = 200)  
    col3.write("<p style = 'text-align: left; font-size:110%; color:#1DA1F2'></p>", unsafe_allow_html = True)


elif choice == "Data Visualization":
    
    visualization_menu = ["Model Comparison" , "20 Most Common Positive-Hashtags Words", "20 Most Common Negative-Hashtags Words",]
    
    choice1 = st.selectbox("Data Vasualization Menu" , visualization_menu )
   
    if choice1 == "20 Most Common Positive-Hashtags Words":
        st.altair_chart(lastmodel.a, use_container_width=True) 
        
    if choice1 == "20 Most Common Negative-Hashtags Words":
       st.altair_chart(lastmodel.b, use_container_width=True)

    if choice1 == "Model Comparison":
        
        column1 , column2 , column3 , column4= st.columns([1, 1.5, 1.5, 1])
        
        F1score = column1.button("  F1 Score  ")
        st.text("")
        
        Accuracy = column2.button('Accuracy Score')
        st.text("")
       
        Precision = column3.button('Precision Score')
        st.text("")
       
        Recall = column4.button('Recall Score')
        st.text("")
     
        
        if F1score:
            st.altair_chart(lastmodel.f1,  use_container_width=False) 
            
        if Accuracy:
            st.altair_chart(lastmodel.acc,  use_container_width=False) 
            
        if Precision:
            st.altair_chart(lastmodel.pre,   use_container_width=False) 
            
        if Recall:
            st.altair_chart(lastmodel.recall,  use_container_width=False) 
