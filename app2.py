from __future__ import unicode_literals
from flask import Flask, render_template, request, redirect, url_for, send_file
import pdfkit
from pytube import YouTube  
import os
import numpy as np
import cv2
import pyautogui
import PIL
from PIL import Image
from skimage import measure
import argparse
import imutils
import pytesseract
import nltk
import subprocess
import pyrebase
import urllib
import re
import shutil
import speech_recognition as sr 
from pydub import AudioSegment
from pydub.utils import make_chunks
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from flask_cors import CORS
import requests

timediff=[]
finalsumm = ""

class Summarize:
    def __init__(self, video_path,video_link):
        self.video_path = video_path
        self.video_link=video_link
        self.new_timestamps = []
        # self.finalsumm = ""

# cosine similarity sumarization
    def read_article(self, file_name):
        file = open(file_name, "r")
        filedata = file.readlines()
        article = filedata[0].split(". ")
        sentences = []

        for sentence in article:
            print(sentence)
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        sentences.pop() 
        
        return sentences

    def sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []
     
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]
     
        all_words = list(set(sent1 + sent2))
     
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
     
        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1
     
        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1
     
        return 1 - cosine_distance(vector1, vector2)
     
    def build_similarity_matrix(self, sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
     
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix


    def generate_summary(self, file_name, top_n=5):
        stop_words = stopwords.words('english')
        summarize_text = []

        # Step 1 - Read text anc split it
        sentences =  self.read_article(file_name)

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = self.build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
        print("Indexes of top ranked_sentence order are ", ranked_sentence)    

        for i in range(top_n):
          summarize_text.append(" ".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize texr
        print("Summarize Text: \n", ". ".join(summarize_text))
        finalsummary = summarize_text
        data = {
          'text': finalsummary
        }
        response = requests.post('http://bark.phon.ioc.ee/punctuator', data=data)
        return response.text

# end of cosine similarity summarization

    def text_recognition(self,file_path):
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        text=""
        words = set(nltk.corpus.words.words())
        source=file_path
        img = cv2.imread(source)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray, img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        gray = cv2.bitwise_not(img_bin)
        kernel = np.ones((2, 1), np.uint8)
        img = cv2.erode(gray, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        out_below = pytesseract.image_to_string(img)
        # cleaned_text = " ".join(w for w in nltk.wordpunct_tokenize(out_below) \
        #     if w.lower() in words or not w.isalpha() and not w.isalnum())
        text=re.sub('[^A-Za-z0-9 ]+', '', out_below)
        words=text.replace("\n","")    
        # split into words by white space
        words = text.split()
        # convert to lower case
        words = [word.lower() for word in words]  
        words = [word for word in words if word.isalpha()]
        
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        text = ' '.join(words)
        return text


    def firebase_authentication(self,file_path,new_timestamps):
        firebaseConfig={"apiKey": "AIzaSyAqxEGopCMFLjyijjadr6ngLXB3jNhs_DY",
        "authDomain": "test-2e3bf.firebaseapp.com",
        "databaseURL": "https://test-2e3bf-default-rtdb.firebaseio.com",
        "projectId": "test-2e3bf",
        "storageBucket": "test-2e3bf.appspot.com",
        "messagingSenderId": "640731267185",
        "appId": "1:640731267185:web:962ed033a14b052be4dbd0",
        "measurementId": "G-SNF25G93N8"}

        firebase=pyrebase.initialize_app(firebaseConfig)
        
        # auth=firebase.auth()
        storage=firebase.storage()
        
        # auth.sign_in_with_email_and_password("dummy@mail.com","123456")
        
        #create new folder in firebase and just append the images that have no duplicates
        path = file_path

        sift = cv2.xfeatures2d.SIFT_create()
        index_params = dict(algorithm=0,trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params,search_params)

        count = 1
        length = len(new_timestamps)
        slides_list=[]
        content=""
        while(count<length-1):
            img1source = path+str(new_timestamps[count])+'.jpg'
            img1 = cv2.imread(img1source)
            original = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)
            original = cv2.resize(original,None,fx=0.4,fy=0.4)
            img2source = path+str(new_timestamps[count+1])+'.jpg'
            img2 = cv2.imread(img2source)
            duplicate = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)
            duplicate = cv2.resize(duplicate,None,fx=0.4,fy=0.4)
            kp_1,desc_1 = sift.detectAndCompute(original,None)
            kp_2,desc_2 = sift.detectAndCompute(duplicate,None)
            if(desc_2 is None or desc_1 is None):
                count = count+1
                continue
            matches = flann.knnMatch(desc_1,desc_2,k=2)
            good_points = []
            for m,n in matches:
                if m.distance < 0.6*n.distance:
                    good_points.append(m)
            no_keyp=0
            if(len(kp_1)<=len(kp_2)):
                no_keyp=len(kp_1)
            else:
                no_keyp=len(kp_2)
            if((len(good_points)/no_keyp*100)<=55):
                #cv2.imwrite(path2+str(new_timestamps[count])+'.jpg',img1) 

                timediff.append(new_timestamps[count][:-4]) #to get list of timestamps

                cloudfilename= 'Slides/'+str(new_timestamps[count])

                cloudfilenameimage='Slides/'+str(new_timestamps[count])+'.jpg'                
                storage.child(cloudfilenameimage).put(img1source)

                content=content+self.text_recognition(img1source)
                content = content+ '\n\n'

                slides_list.append(cloudfilename)
            count=count+1

        timediff.append(new_timestamps[count][:-4]) #to get list of timestamps

        cloudfilename= 'Slides/'+str(new_timestamps[count])

        cloudfilenameimage='Slides/'+str(new_timestamps[count])+'.jpg'                
        storage.child(cloudfilenameimage).put(img1source)

        content=content+self.text_recognition(img1source)
        content = content+ '\n\n'

        slides_list.append(cloudfilename)
        cloudfilenametext='Slides/Content.txt'
        textfilepath=path+'Content.txt'
        f = open(textfilepath, "w")
        f.write('\n'+content)
        f.close()
        storage.child(cloudfilenametext).put(textfilepath)
        # slidess = ' '.join(slides_list) 
        shutil.rmtree(path)    
        return slides_list 
        
     
    def timestamps_assigning(self,timestamp,folder_path):
        #renaming the timestamps to a proper format
        timestamp=list(str(timestamp).split('\n'))
        new_timestamps = []
        for i in timestamp:
            t1=i.replace('.',':')
            t2=t1.replace(':','-')
            t2=t2[0:-3]
            new_timestamps.append(t2)
        new_timestamps=new_timestamps[0:-1]
        
        #renaming the images with their timestamps
        i=0
        path = folder_path+'\\'
        path = path.replace('\\','/')

        for filename in os.listdir(path):
            if i<len(new_timestamps):
                dest=path+str(new_timestamps[i])+'.jpg'
                source=path+filename
                os.rename(source,dest)
                i=i+1
            else:
                #ffprobe did not find a timestamp for this frame
                os.remove(path+filename)
        
        return self.firebase_authentication(path,new_timestamps)

    def audiototext(self,filepath):
        firebaseConfig={"apiKey": "AIzaSyAqxEGopCMFLjyijjadr6ngLXB3jNhs_DY",
        "authDomain": "test-2e3bf.firebaseapp.com",
        "databaseURL": "https://test-2e3bf-default-rtdb.firebaseio.com",
        "projectId": "test-2e3bf",
        "storageBucket": "test-2e3bf.appspot.com",
        "messagingSenderId": "640731267185",
        "appId": "1:640731267185:web:962ed033a14b052be4dbd0",
        "measurementId": "G-SNF25G93N8"}

        firebase=pyrebase.initialize_app(firebaseConfig)
        storage=firebase.storage()

        ffmpeg = 'C:/Users/Amritha/Desktop/fyp_code/ffmpeg.exe'
        command2mp3 = ffmpeg +" -i "+ filepath +" speech.mp3"
        command2wav = ffmpeg +" -i speech.mp3 speech.wav"
        os.system(command2mp3)
        os.system(command2wav)

        audio = AudioSegment.from_wav("speech.wav")  
        timestamp_length = []

        listoftexts = []
        # listoftexts.append("FIRST SLIDE")

        folder_name = "audio-chunks"
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        whole_text = ""
        text_file = open("speech.txt", "w")

        r = sr.Recognizer()

        timediff1 = list(set(timediff))
        timediff1.sort()        
        print(timediff1)

        i=0
        while i<len(timediff1)-1:
            print(timediff1[i])
            print(timediff1[i+1])
            fmt = '%H-%M-%S'
            tstamp1 = datetime.strptime(timediff1[i], fmt)
            tstamp2 = datetime.strptime(timediff1[i+1], fmt)
            td = tstamp2 - tstamp1
            td_mins = int(td.total_seconds())*1000 
            if(td_mins>0):
                print('The difference is approx. %s milliseconds' % td_mins)
                timestamp_length.append(td_mins)
            i=i+1
        print(timestamp_length)
        
        start = 0
        j = 0
        k = 0
        for  idx,t in enumerate(timestamp_length):
            #break loop if at last element of list
            if idx == len(timestamp_length):
                break
            end = start + t 
            print("split at [ {}:{}]".format(start, end))
            audio_chunk=audio[start:end]

            if end-start<59000:
                chunk_filename = os.path.join(folder_name, f"{start}to{end}.wav")
                audio_chunk.export( chunk_filename, format="wav")
                with sr.AudioFile(chunk_filename) as source:
                    audio_listened = r.record(source)
                try:
                    text = r.recognize_google(audio_listened)
                except sr.UnknownValueError as e:
                    print("Error:", str(e))
                else:
                    text = f"{text.capitalize()}. "
                    data = {
                      'text': text
                    }
                    response = requests.post('http://bark.phon.ioc.ee/punctuator', data=data)
                    listoftexts.append(response.text)
                    # text_file.write("%s to %s:\n %s\n" %(timediff[j],timediff[j+1],text))
                    if j<len(timediff):
                        j=j+1
                    print(response.text)
                    whole_text += text
            else:
                chunk_length = 59000 
                chunks = make_chunks(audio_chunk, chunk_length)
                # text_file.write("%s to %s:\n" %(timediff[j],timediff[j+1]))
                if j<len(timediff):
                    j=j+1
                for i,small_chunk in enumerate(chunks, start=1):
                    chunk_filename = os.path.join(folder_name, f"{start}to{end}-{i}.wav")
                    small_chunk.export(chunk_filename, format="wav")
                    with sr.AudioFile(chunk_filename) as source:
                        audio_listened = r.record(source)
                    try:
                        text = r.recognize_google(audio_listened)
                    except sr.UnknownValueError as e:
                        print("Error:", str(e))
                    else:
                        text = f"{text.capitalize()}. "
                        data = {
                          'text': text
                        }
                        response = requests.post('http://bark.phon.ioc.ee/punctuator', data=data)
                        listoftexts.append(response.text)
                        # text_file.write(text)
                        print(response.text)
                        whole_text += text
            start = end

        
        print(whole_text)
        text_file.write(whole_text)
        text_file.close()
        path_on_cloud = "Slides/audiototext.txt"
        path_local = "speech.txt"
        storage.child(path_on_cloud).put(path_local)

        # summary
        finalsumm = self.generate_summary("speech.txt", 2)
        print("finalsumm:")
        print(finalsumm)
        print("listoftexts:")





        print(listoftexts)
        return listoftexts



    def get_iframes_from_downloaded_video(self,video_path):
        inFile = video_path
        oString = 'frame'
        

        ffmpeg = 'C:/Users/Amritha/Desktop/fyp_code/ffmpeg.exe'
        outDir = os.getcwd()
        
        newFolder=outDir+"\Iframes"
        os.mkdir(newFolder)
        
        outDir = outDir + "\\" + "Iframes" + "\\" + oString + '%04d.jpg'
        outFile = outDir.replace('\\','/')
    
        cmd = [ffmpeg,'-i', inFile,'-vf', "select='eq(pict_type,PICT_TYPE_I)'",'-vsync','vfr',outFile]
        subprocess.call(cmd)
        
        ffprobe = 'C:/Users/Amritha/Desktop/fyp_code/ffprobe.exe'
        cmd=ffprobe + ' -v error -skip_frame nokey -show_entries frame=pkt_pts_time -select_streams v -of csv=p=0 '+inFile+' -sexagesimal'
        
        timestamp = os.popen(cmd).read()
        return self.timestamps_assigning(timestamp,newFolder)
    
    def get_iframes_from_youtube_video(self,video_link):
        path = os.getcwd() 
        try:
            yt_obj = YouTube(video_link)
    
            filters = yt_obj.streams.filter(progressive=True, file_extension='mp4')
    
            # download the highest quality video
            filters.get_highest_resolution().download(output_path=path,filename='Input')
            #print('Video Downloaded Successfully')
        except Exception as e:
            print(e)
        inFile = path+'\Input.mp4'
        inFile = inFile.replace('\\','/')
        oString = 'frame'
        
        ffmpeg = 'C:/Users/Amritha/Desktop/fyp_code/ffmpeg.exe'
        outDir = os.getcwd()
        
        newFolder=outDir+"\Iframes"
        os.mkdir(newFolder)
        
        outDir = outDir + "\\" + "Iframes" + "\\" + oString + '%04d.jpg'
        outFile = outDir.replace('\\','/')
    
        cmd = [ffmpeg,'-i', inFile,'-vf', "select='eq(pict_type,PICT_TYPE_I)'",'-vsync','vfr',outFile]
        subprocess.call(cmd)
        
        ffprobe = 'C:/Users/Amritha/Desktop/fyp_code/ffprobe.exe'
        cmd=ffprobe + ' -v error -skip_frame nokey -show_entries frame=pkt_pts_time -select_streams v -of csv=p=0 '+inFile+' -sexagesimal'
        
        timestamp = os.popen(cmd).read()
        return self.timestamps_assigning(timestamp,newFolder)

    def download_video(self,video_path):
        path = os.getcwd() 
        try:
            yt_obj = YouTube(video_link)
    
            filters = yt_obj.streams.filter(progressive=True, file_extension='mp4')
    
            # download the highest quality video
            filters.get_highest_resolution().download(output_path=path,filename='Input')
            #print('Video Downloaded Successfully')
        except Exception as e:
            print(e)
        inFile = path+'\Input.mp4'
        return inFile

    def get_slide_text(self):
        firebaseConfig={"apiKey": "AIzaSyAqxEGopCMFLjyijjadr6ngLXB3jNhs_DY",
        "authDomain": "test-2e3bf.firebaseapp.com",
        "databaseURL": "https://test-2e3bf-default-rtdb.firebaseio.com",
        "projectId": "test-2e3bf",
        "storageBucket": "test-2e3bf.appspot.com",
        "messagingSenderId": "640731267185",
        "appId": "1:640731267185:web:962ed033a14b052be4dbd0",
        "measurementId": "G-SNF25G93N8"}

        firebase=pyrebase.initialize_app(firebaseConfig)
        # auth=firebase.auth()
        storage=firebase.storage()
            
        return(storage.child("Slides/Content.txt").get_url(None))

    def get_audio_text(self):
        firebaseConfig={"apiKey": "AIzaSyAqxEGopCMFLjyijjadr6ngLXB3jNhs_DY",
        "authDomain": "test-2e3bf.firebaseapp.com",
        "databaseURL": "https://test-2e3bf-default-rtdb.firebaseio.com",
        "projectId": "test-2e3bf",
        "storageBucket": "test-2e3bf.appspot.com",
        "messagingSenderId": "640731267185",
        "appId": "1:640731267185:web:962ed033a14b052be4dbd0",
        "measurementId": "G-SNF25G93N8"}

        firebase=pyrebase.initialize_app(firebaseConfig)
        # auth=firebase.auth()
        storage=firebase.storage()
            
        return(storage.child("Slides/audiototext.txt").get_url(None))


class Webpage:
    def __init__(self):
        self.app = Flask(__name__,template_folder='.')
        @self.app.route('/')
        def index():
            return render_template("main2.html")

        @self.app.route('/upload', methods=['POST'])
        def getfilepath():
            video_path = request.form['videopath']
            video_link = request.form['videolink']
            listslides = []
            summary=Summarize(video_path,video_link)
            # print(video_link)
            # print(video_path)
            if(video_link==""):
                listslides=summary.get_iframes_from_downloaded_video(video_path)
                listtext = summary.get_slide_text()
                listaudio = summary.audiototext(video_path)
                audiotext = summary.get_audio_text()
                summcontent = summary.generate_summary("speech.txt",2)
            else:
                listslides=summary.get_iframes_from_youtube_video(video_link)
                listtext = summary.get_slide_text()
                video = summary.download_video(video_link)
                listaudio = summary.audiototext(video)
                audiotext = summary.get_audio_text()
                summcontent = summary.generate_summary("speech.txt",2)
            
            return render_template('upload.html', l = listslides, lt = listtext, aud = audiotext, la = listaudio, summ = summcontent)

        @self.app.route('/download')
        def download_file():
            print("hell0")
            path = "C:/Users/Amritha/Downloads/samplex.pdf"
            pdfkit.from_url('http://127.0.0.1:5000/upload',"C:/Users/Amritha/Downloads/samplex.pdf") 
            return send_file(path,as_attachment=True)

        # @self.app.route('/display', methods = ['POST'])
        # def display():
        #     return render_template('random.html',var=['moo','meow'])

if __name__ == "__main__":
    web = Webpage()
    web.app.run(debug=True)