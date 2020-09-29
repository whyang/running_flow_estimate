# -*- coding: utf-8 -*-
"""
Created on Sep. 29, 2020
使用馬拉松賽事跑者人流分析與預測模型，進行預測
Raw data: 田中馬拉松(2017年、2018年、2019年)，全馬組跑者成績紀錄，建立之預測模型
Input: 感應點距離(離起點)、馬拉松舒適指數(環境因子、體感因子等9項)、
        觀測時間(起跑後通過某個感應點時間(秒), delta_time, i.e., 預測時間(成績))、
        全馬參賽(或報名)人數、鳴槍時間(秒)
Output: 預測人數(某個感應點，目前鳴槍時間(秒)後
@author: Wen-Hsin Yang
"""

import os
import pandas as pd
import numpy as np
import time
import joblib #pkl模型 format
from flask import Flask, render_template, request # , url_for

def revRunningPerformanceToTime(delta=0, base=0):
    timeStamp = delta + base
    #timeStamp //= 1e3
    timearr = time.gmtime(timeStamp) #由於沒有指定timezone，原先之時間戳轉換為utc時間的struct_time會主動調整為(utc+8)，使用gmtime會調整本地時間差8個小時
    otherStyleTime = time.strftime("%H:%M:%S", timearr)
    #print('*** 跑步成績 (現在觀測時間) = ', timeStamp, '(sec.)', otherStyleTime)
    return otherStyleTime

num_runner = 0
base_dir = os.path.dirname(__file__)   
app = Flask(__name__, 
            static_folder=os.path.join(base_dir, 'static'),
            template_folder=os.path.join(base_dir, 'templates'))

@app.route('/')
def home():
	return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        #speed = request.values['speed']
        distance = request.values['distance']
        score = request.values['score']
        temp = request.values['temp']
        hum = request.values['hum']
        heatindex = request.values['heatindex']
        AQI = request.values['AQI']
        PM = request.values['PM']
        WR = request.values['WR']
        rain = request.values['rain']
        runner = request.values['runner']
        commence_time = request.values['commence_time']

        ###
        # normalize input features
        ##   
        distance_ = float(distance) / range_dist
        score_ = float(score) / range_runscore
        speed_ = ((float(distance) / 1000) / (float(score) / 3600)) / range_speed
        temp_ = float(temp) / range_temp
        hum_ = float(hum) / range_hum
        heatindex_ = float(heatindex) / range_hi
        AQI_ = float(AQI) / range_aqi
        PM_ = float(PM) / range_pm
        WR_ = float(WR) / range_wr
        rain_ = float(rain) / range_rain
        runner = float(runner)
        num_runner = runner
        commence_time = float(commence_time)
        range_commencetime = int(commence_time)
        
        pred = np.array([[speed_, distance_, temp_, hum_, heatindex_, AQI_, PM_, WR_, rain_, score_]])                          
        x_pred = pd.DataFrame(data=pred, 
                              columns=['速度', '距離', '溫度', '濕度', '熱中暑危險係數',
                                       '空氣品質指標', '細懸浮微粒', '蒲福風級', '小時雨量', '預測時間'])
        print('x_pred = ', x_pred)  
               
        # get the prediction results (random forest regression) respecting to the input x_pred
        y_pred = rf_reg.predict(x_pred)
        print('model_random forest regression')
        print('runner_flow = {a} '.format(a=y_pred))
        print('runner_flow = {a} '.format(a=y_pred * num_runner))
        
        # get the prediction results (XGBoosting regression) respecting to the input x_pred    
        y_pred_1 = xgboost_reg.predict(x_pred)
        print('model_xgboost regression')
        print('runner_flow = {a} '.format(a=y_pred_1))
        print('runner_flow = {a} '.format(a=y_pred_1 * num_runner))
                
        # print out the daily time
        print('print out (XGBoosting regression')
        output_flow = int(y_pred_1 * num_runner)
        score_time = score_ * range_runscore
        output_score = revRunningPerformanceToTime(delta=score_time, base=range_commencetime)
        print('output flow={a}, time={b} {c}'.format(a=output_flow, b=int(score_time), c=output_score))
        output_commence_time = revRunningPerformanceToTime(delta=range_commencetime)
        print('鳴槍時間={a}'.format(a=output_commence_time))
        
        #
    # send out the prediction results and their corresponding inputs of features
    return render_template('output.html', 
                           prediction_flow=output_flow, 
                           distance=distance, #x_pred.loc[0,'距離'],
                           prediction_score=output_score,
                           temp=temp, #x_pred.loc[0,'溫度'], 
                           hum=hum, #x_pred.loc[0,'濕度'],
                           heatindex=heatindex, #x_pred.loc[0,'熱中暑危險係數'], 
                           AQI=AQI, #x_pred.loc[0,'空氣品質指標'],
                           PM=PM, #x_pred.loc[0,'細懸浮微粒'], 
                           WR=WR, #x_pred.loc[0,'蒲福風級'], 
                           rain=rain,
                           runner=int(runner),
                           commence_time=output_commence_time)
# end of def predict():
    
if __name__ == '__main__':
    # load existing model: currently is based on the Tianzhong Marathon 2017~2019
    print('** load the existing model  (joblib)')
    model_dir = os.path.join(base_dir, 'model') # the path for storing models
    model = os.path.join(model_dir, 'rf_reg_flow(joblib).pkl') 
    print('model = {a}'.format(a=model))
    rf_reg = joblib.load(model)
    
    # for testing
    model = os.path.join(model_dir, 'xgboost_reg_flow(joblib).pkl') 
    print('model = {a}'.format(a=model))
    xgboost_reg = joblib.load(model)

    
    # read into configuration parameter (i.e., the range of each features 
    config = os.path.join(model_dir, 'config.xlsx') # the configuration file
    df = pd.read_excel(config,
                       usecols=['鳴槍時間', '速度', '距離', '溫度', '濕度', '熱中暑危險係數',
                                '空氣品質指標', '細懸浮微粒', '蒲福風級', '小時雨量',
                                '預測時間', '預測人數(最多)', '全馬參賽(或報名)人數'])        

    # display values in the configuration file     
    range_commencetime = df.loc[0, '鳴槍時間']
    print('commence_time = ', range_commencetime)
    range_speed = df.loc[0, '速度'] 
    print('range_speed = ', range_speed)
    range_dist = df.loc[0, '距離']
    print('range_dist = ', range_dist)
    range_runscore = df.loc[0, '預測時間']
    print('range_runscore = ', range_runscore)
    range_temp = df.loc[0, '溫度']
    print('range_temp = ', range_temp)
    range_hum = df.loc[0, '濕度']
    print('range_hum = ', range_hum)
    range_hi = df.loc[0, '熱中暑危險係數']
    print('range_hi = ', range_hi)
    range_aqi = df.loc[0, '空氣品質指標']
    print('range_aqi = ', range_aqi)
    range_pm = df.loc[0, '細懸浮微粒']
    print('range_pm = ', range_pm)
    range_wr = df.loc[0, '蒲福風級']
    print('range_wr = ', range_wr)
    range_rain = df.loc[0, '小時雨量']
    print('range_rain = ', range_rain)
    range_runflow = df.loc[0, '預測人數(最多)'] # maximal number of runners in the training data 
    print('range_runflow = ', range_runflow)
    num_runner = df.loc[0, '全馬參賽(或報名)人數'] # 預測該場全馬賽事之參賽(或報名)人數
    print('num_runner = ', num_runner)

    # initiate web framework    
    # app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=False, host='0.0.0.0', port=5000) 
    # setting debug=False for enabling printing info. in flask

###
# end of file
##