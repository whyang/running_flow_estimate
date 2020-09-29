# -*- coding: utf-8 -*-
"""
Created on Sep. 29, 2020
建立馬拉松賽事跑者人流分析與預測模型
Raw data: 田中馬拉松(2017年、2018年、2019年)，全馬組跑者成績紀錄
Feature: 跑速、感應點距離(離起跑點)、馬拉松舒適指數(環境因子、體感因子等9項)、
        預測成績(起跑後通過某個感應點時間(秒), delta_time)(輸入特徵值進行正規化處理)
Output: 預測人數
@author: Wen-Hsin Yang
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import joblib #pkl模型 format
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#from mpl_toolkits.mplot3d import Axes3D

# Register converters to avoid warnings
pd.plotting.register_matplotlib_converters()
plt.rc("figure", figsize=(16,12))
plt.rc("font", size=16)
plt.rcParams['axes.unicode_minus'] = False # 修復負號顯示問題(正黑體)

#####################
# declare functions #
#####################
##
# remove leading and trailing characters of each value across all cells in dataframe
def trim_all_cells(df):
    # trim whitespace from ends of each value across all series in dataframe
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.applymap(trim_strings)

def heatmap(x, y, size, corr):
    ###
    # heatmap 1: demonstrate the correlation of each two features in terms of the size of correlated ratio (position/negative)
    ##
    fig, ax = plt.subplots(figsize=(16,12))
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]

    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)}
    
    #sns.set(font=['sans-serif'])
    size_scale = 300
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_title('跑拉松跑者人流分析預測(Feature Correlation)')
    ax.set_xlabel('特徵')
    ax.set_ylabel('特徵')
    plt.show() # display the graph
    # save figure to the file
    figfile = os.path.join(figpath, 'Feature Correlation_1.jpg')
    fig.savefig(figfile) # save the graph into a file 
    
    ###
    # heatmap 2: demonstrate the correlation of each two features in terms of the correlated ratio
    ##
    fig, ax1 = plt.subplots(figsize=(16,12))
    corr = corr.pivot('x', 'y', 'value')
    ax1 = sns.heatmap(corr, vmax=1, vmin=-1, cmap='coolwarm', center=0, robust=True,
                     annot=True, annot_kws={'size':16}, fmt='.1f',
                     linewidths=0.5, square=True)
    ax1.set_xticklabels(ax1.get_yticklabels(), rotation=45, fontsize=16)
    ax1.set_title('跑拉松跑者人流分析預測(Feature Correlation)')
    ax1.set_xlabel('特徵 feature')
    ax1.set_ylabel('特徵 feature')
    plt.show()
    # save figure to the file
    figfile = os.path.join(figpath, 'Feature Correlation_2.jpg')
    fig.savefig(figfile) # save the graph into a file 
    
################
# main program #
################
if __name__ == '__main__':
    ###
    # initialize environment's configuration
    ##
    # home directory
    base_dir = os.path.dirname(__file__)
    # directory for training data
    datapath = os.path.join(base_dir, 'data')
    if not os.path.isdir(datapath):
        os.mkdir(datapath)
    # directory for storing the figurer that are run by the trained model
    figpath = os.path.join(base_dir, 'figure')
    if not os.path.isdir(figpath):
        os.mkdir(figpath)
    # directory for storing the trainned model
    savepath = os.path.join(base_dir, 'model')
    if not os.path.isdir(savepath):
        os.mkdir(savepath)  
    
    ###
    # step 1: conduct data preprocessing
    ##
    # read into the dataset   
    datafile = os.path.join(datapath, 'dataset.xlsx')
    df = pd.read_excel(datafile,
                       usecols=['預測人數', '預測時間',
                                '速度', '距離', '溫度', '濕度', '熱中暑危險係數',
                                '空氣品質指標', '細懸浮微粒', '蒲福風級', '小時雨量', '比例'])
    trim_all_cells(df) # remove leading and tailing white space of string (content of cell in dataframe)
    df.dropna(inplace=True) # omit the row of data in which any NAN value is contained            
 
    # normalized by using MinMax standarization for features (in the dataframe X)
    X = pd.DataFrame()
    sc1 = MinMaxScaler() # range 0~1
        
    # 跑步速度 (0~22 km/h)
    sc = MinMaxScaler(feature_range=(0, 22))
    x_speed1 = pd.DataFrame(sc.fit_transform(df[['速度']].astype(float)))
    x_speed = pd.DataFrame(sc1.fit_transform(x_speed1[:].astype(float)))
    x_speed.columns = ['速度']
        
    # 感應點位置(距離) (0~43,000 m)
    sc = MinMaxScaler(feature_range=(0, 43000))
    x_dist1 = pd.DataFrame(sc.fit_transform(df[['距離']].astype(float)))
    x_dist = pd.DataFrame(sc1.fit_transform(x_dist1[:].astype(float)))
    x_dist.columns = ['距離']        
        
    # 體感因子(溫度) (0~40 度C)
    sc = MinMaxScaler(feature_range=(0, 40))
    x_temp1 = pd.DataFrame(sc.fit_transform(df[['溫度']].astype(float)))
    x_temp = pd.DataFrame(sc1.fit_transform(x_temp1[:].astype(float)))
    x_temp.columns = ['溫度']
        
    # 體感因子(濕度)(0~100 %)
    sc = MinMaxScaler(feature_range=(0, 100))
    x_hum1 = pd.DataFrame(sc.fit_transform(df[['濕度']].astype(float)))
    x_hum = pd.DataFrame(sc1.fit_transform(x_hum1[:].astype(float)))
    x_hum.columns = ['濕度']
        
    # 體感因子(熱中暑危險係數 Heat Index)(0~100 度C)
    sc = MinMaxScaler(feature_range=(0, 100))
    x_hi1 = pd.DataFrame(sc.fit_transform(df[['熱中暑危險係數']].astype(float)))
    x_hi = pd.DataFrame(sc1.fit_transform(x_hi1[:].astype(float)))
    x_hi.columns = ['熱中暑危險係數']
        
    # 環境因子(空氣品質指標 AQI) (0~200)
    sc = MinMaxScaler(feature_range=(0, 200))
    x_aqi1 = pd.DataFrame(sc.fit_transform(df[['空氣品質指標']].astype(float)))
    x_aqi = pd.DataFrame(sc1.fit_transform(x_aqi1[:].astype(float)))
    x_aqi.columns = ['空氣品質指標']
        
    # 環境因子(細懸浮微粒 PM2.5) (0~72)
    sc = MinMaxScaler(feature_range=(0, 72))
    x_pm1 = pd.DataFrame(sc.fit_transform(df[['細懸浮微粒']].astype(float)))
    x_pm = pd.DataFrame(sc1.fit_transform(x_pm1[:].astype(float)))
    x_pm.columns = ['細懸浮微粒']
        
    # 環境因子(蒲福風級) (0~5 級風)
    sc = MinMaxScaler(feature_range=(0, 5))
    x_wr1 = pd.DataFrame(sc.fit_transform(df[['蒲福風級']].astype(float)))
    x_wr = pd.DataFrame(sc1.fit_transform(x_wr1[:].astype(float)))
    x_wr.columns = ['蒲福風級']
        
    # 環境因子(小時雨量) (0~41 mm/h)
    sc = MinMaxScaler(feature_range=(0, 41))
    x_hr1 = pd.DataFrame(sc.fit_transform(df[['小時雨量']].astype(float)))
    x_hr = pd.DataFrame(sc1.fit_transform(x_hr1[:].astype(float)))
    x_hr.columns = ['小時雨量']

    # 感應點觀測時間(在某個感應點特定的時間，觀察之後每10分鐘持續1小時，通過感應的人數) (0~關門時間 sec.)
    sc = MinMaxScaler() #MinMaxScaler(feature_range=(0, 21600))
    x_score1 = pd.DataFrame(sc.fit_transform(df[['預測時間']].astype(float)))
    x_score = pd.DataFrame(sc1.fit_transform(x_score1[:].astype(float)))
    x_score.columns = ['預測時間']
    range_runscore = sc.data_max_ # keep the maximal value in the configuration
        
    # combine all features as the dataframe X
    X = pd.concat([x_speed, x_dist, x_temp, x_hum, x_hi, x_aqi, x_pm, x_wr, x_hr, x_score], axis=1)
    # assign columns' labels into the training dataset
    X.columns = ['速度', '距離', '溫度', '濕度', '熱中暑危險係數', 
                 '空氣品質指標', '細懸浮微粒', '蒲福風級', '小時雨量', '預測時間']

    # applying MinMax scheme to normalize the prediction factors (i.e., the runners' number w.r.t. running flow and runners' scores)
    y = pd.DataFrame()
    # forecasted number of runners in a flow
    sc = MinMaxScaler() 
    y_flow1 = pd.DataFrame(sc.fit_transform(df[['預測人數']].astype(float)))
    y_flow = pd.DataFrame(sc1.fit_transform(y_flow1[:].astype(float)))
    y_flow.columns = ['預測人數']
    range_runflow = sc.data_max_ # keep the maximal number in the configuration 
    # establish y by concatting flow_number and scores
    y = y_flow #pd.concat([y_flow, y_score], axis=1)
        
    y = sc.fit_transform(df[['比例']])
    
    num_runner = 3854 # 預測該場全馬賽事之參賽(或報名)人數, default=3,854
    
    # keep the normalization parameters of the features in the configuration file
    config_df = pd.DataFrame(
        {'鳴槍時間' : 21600, # 6 a.m.
         '速度' : 22, # range 0~22
         '距離' : 43000, # range 0~43,000
         '溫度' : 40, # range 0~40
         '濕度' : 100, # range 0~100
         '熱中暑危險係數' : 100, # range 0~100
         '空氣品質指標' : 200, # range 0~200
         '細懸浮微粒' : 72, # range 0~72
         '蒲福風級' : 5, # range 0~5
         '小時雨量' : 41, # range 0~41 mm/hr
         # instead of 預測時間, we rename to 觀測時間 on the specific Station due to fitting the usage scenario
         '預測時間' : range_runscore, # range 0~range_runsore sec. 關門時間,
         '預測人數(最多)' : range_runflow, # maximal number of runners in the training data 
         '全馬參賽(或報名)人數' : num_runner # 預測該場全馬賽事之參賽(或報名)人數
        })
    configfile = os.path.join(savepath, 'config.xlsx')
    config_df.to_excel(configfile, index=False, encoding='cp950')
                
    ###
    # step 2: split training and testing datasets
    ##
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
    ###
    # step 3: train a model (univariate random forest regression)
    ##
    print('----------------------- RandomForestRegressor')
        
    # generate random forest regression model
    rf_reg_flow = RandomForestRegressor()
    rf_reg_flow.fit(X_train, y_train.ravel()) 
        
    # evaluate model (univariate random forest regression)
    y_flow_pred = rf_reg_flow.predict(X_test)  
    mse = mean_squared_error(y_test, y_flow_pred) * num_runner
    rmse = math.sqrt(mse)
    print("MSE, RMSE: %f %f" % (mse, rmse))
    print('score= {a}'.format(a=rf_reg_flow.score(X_test, y_test)))
    print('importance= {a}'.format(a=rf_reg_flow.feature_importances_))
        
    # visualize the prediction result: 觀測時間 vs 預測人數
    # input: '觀測時間' in X-axis
    # output: '預測人數' in Y-axis
    plt.rc("figure", figsize=(16,10))
    plt.rc("font", size=16)
    plt.rcParams['axes.unicode_minus'] = False # 修復負號顯示問題(正黑體)
    fig, ax = plt.subplots(figsize=(16, 12))     
    # real results in the testing data
    X1 = df.loc[X_test.index] # take back original dataframe values w.r.t X.test which have conducted normalization 
    ax.scatter(X1['預測時間'], y_test * num_runner, c='blue', marker='s', alpha=0.6, label='田中馬_全馬組(2017~2019年)紀錄')
    #ax.scatter(X_test['預測時間'], y_test, c='blue', marker='s', alpha=0.6, label='田中馬_全馬組(2017~2019年)紀錄')
    # predicted results w.r.t. testing data
    ax.scatter(X1['預測時間'], y_flow_pred * num_runner, c='red', marker='x', label='預測模型(Random Forest Regression)')
    #ax.scatter(X_test['預測時間'], y_flow_pred, c='red', marker='x', label='預測模型(Random Forest Regression)')
    ax.set_xlabel('觀測時間 (單位:秒)', fontsize=16) 
    ax.set_ylabel('人數 (單位:人)', fontsize=16)    
    ax.legend()
    ax.grid(True)
    plt.show() # display the graph
    # save figure to the file
    figfile = os.path.join(figpath, '預估人流_RF.jpg')
    fig.savefig(figfile) # save the graph into a file 

    ###
    # step 4: keep training model
    ##        
    # save model using joblib package
    savefile = os.path.join(savepath, 'rf_reg_flow(joblib).pkl')
    joblib.dump(rf_reg_flow, savefile) 
    '''
    #check saved model
    rf_reg_1_flow = joblib.load(savefile)
    print('** save model then checking for it (rf_reg_flow)')
    print(rf_reg_1_flow.predict(X_test)       
    '''
    
    ###
    # step 5: observe the correlation between any two features
    ##
    columns = ['速度', '距離', '溫度', '濕度', '熱中暑危險係數', '空氣品質指標', 
               '細懸浮微粒', '蒲福風級', '小時雨量', '預測時間', '預測人數', '比例']      
    corr = df[columns].corr()
    corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']
    heatmap(x=corr['x'], y=corr['y'], size=corr['value'].abs(), corr=corr)
        
    ##
    # evaluate XGBoost reg:squarederror
    ##
    print('----------------------- XGBRegressor')
    xgb_reg = xgb.XGBRegressor(objective = 'reg:squarederror',
                               #learning_rate = 0.1,
                               #max_depth = 30,
                               #subsample = 0.5,
                               #colsample_bytree = 0.5,
                               #alpha = 0.1,
                               n_estimators = 5000)
    xgb_reg.fit(X_train, y_train)
    y_flow_pred = xgb_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_flow_pred) * num_runner
    rmse = math.sqrt(mse)
    print("MSE, RMSE: %f %f" % (mse, rmse))
        
    # save model using joblib package
    savefile = os.path.join(savepath, 'xgboost_reg_flow(joblib).pkl')
    joblib.dump(xgb_reg, savefile)
    '''
    #check saved model
    xgb_reg_1_flow = joblib.load(savefile) 
    print('** save model then checking for it (xgboost_reg_flow)')
    print(xgb_reg_1_flow.predict(X_test)*num_runner)           
    '''
    
    #plt.rc("figure", figsize=(16, 10))
    #plt.rc("font", size=16)
    #plt.rcParams['axes.unicode_minus'] = False # 修復負號顯示問題(正黑體)       
 
    xgb.plot_tree(xgb_reg, num_trees=12)
    xgb.plot_importance(xgb_reg)
    plt.show()
    
    # visualize the prediction result: 觀測時間 vs 預測人數
    # input: '觀測時間' in X-axis
    # output: '預測人數' in Y-axis
    plt.rc("figure", figsize=(16,12))
    plt.rc("font", size=16)
    plt.rcParams['axes.unicode_minus'] = False # 修復負號顯示問題(正黑體)
    fig, ax = plt.subplots(figsize=(16, 12))     
    # real results in the testing data
    X1 = df.loc[X_test.index] # take back original dataframe values w.r.t X.test which have conducted normalization 
    ax.scatter(X1['預測時間'], y_test * num_runner, c='blue', marker='s', alpha=0.6, label='田中馬_全馬組(2017~2019年)紀錄')
    #ax.scatter(X_test['預測時間'], y_test, c='blue', marker='s', alpha=0.6, label='田中馬_全馬組(2017~2019年)紀錄')
    # predicted results w.r.t. testing data
    ax.scatter(X1['預測時間'], y_flow_pred * num_runner, c='red', marker='x', label='預測模型(XGBoost Regression)')
    #ax.scatter(X_test['預測時間'], y_flow_pred, c='red', marker='x', label='預測模型(XGBoost Regression)')
    ax.set_xlabel('觀測時間 (單位:秒)', fontsize=16) 
    ax.set_ylabel('人數 (單位:人)', fontsize=16)    
    ax.legend()
    ax.grid(True)
    plt.show() # display the graph
    # save figure to the file
    figfile = os.path.join(figpath, '預估人流_XGBoost.jpg')
    fig.savefig(figfile) # save the graph into a file 

###############
# end of file #
###############  