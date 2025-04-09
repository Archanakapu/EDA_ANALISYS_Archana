import os
os.environ['GEMINI_API_KEY']='AIzaSyBN1wMuBIJ4dYHnep2VkoNcEZxgzZ1kXts'
#pip install -q -U google-generativeai

import google.generativeai as genai
genai.configure(api_key="AIzaSyBN1wMuBIJ4dYHnep2VkoNcEZxgzZ1kXts")
model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Explain how AI works")
print(response.text)