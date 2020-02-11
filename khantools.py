def report1 (dataframe,n_highest_counts):
    ''' Returns a dataframe reporting on the value counts of input frame,
    for the top n_highest_counts values'''
    
    master={}
        
    for column in dataframe.columns:
        
        master[column]={}
        col_dict = master[column]
        col_dict['type'] = str(dataframe[column].dtypes)
        col_dict['% empty'] = round(((len(dataframe)-dataframe[column].count())/len(dataframe))*100,2)
        col_dict['unique values'] = dataframe[column].nunique()
        
        x = 1
        series1 = dataframe[column].value_counts().head(n_highest_counts)
        series1 = round((series1/len(dataframe)) * 100, 2)        
        
        for index,item in series1.items():
            value_prop = str(x) + 'nth_value_%'
            value_name = str(x) + 'nth_value'
            col_dict[value_name] = index
            col_dict[value_prop] = item
            x += 1
        
    df_report=pd.DataFrame.from_dict(master,orient='index')
    df_report.sort_values(['1nth_value_%'],ascending=False,inplace=True)
        
    return df_report

def row_loss(df_base,df_current):
    ''' Returns list describing difference in row counts between two input frames'''
    
    rows_dropped = len(df_base) - len(df_current)
    rows_left = len(df_current)
    row_loss_perc = ((len(df_base) - rows_dropped)/len(df_base)) * 100
    row_loss_perc = round(row_loss_perc,2)
    
    metrics = [rows_dropped,rows_left,row_loss_perc]
    
    return metrics

def loss_report (df_base,df_current):
    ''' Prints row loss report '''
    stats_list = row_loss(df_base,df_current)
    
    for x in stats_list:
        x = str(x)
      
    string = "Rows Dropped: {}    Rows Left: {}   Percentage Remaining: {}".format(stats_list[0],stats_list[1],stats_list[2])
    
    print(string)

def duplicates_list (dataframe,column):
    '''Returns list of index labels of duplicate column values'''
    
    df_id_Counts = dataframe[column].value_counts()
    repeats = df_id_Counts[df_id_Counts > 1]
    repeats = list(repeats.index)
    
    return repeats

def box_matrix (dataframe):
    ''' generates boxplots for all columns in dataframe'''
    df = dataframe
    col_nums = len(df.columns)
    if col_nums % 7 > 0:
        chart_cols = int(round((col_nums/7)+1,0))
    else:
        chart_cols = int(round((col_nums/7),0))

    figure, ax = plt.subplots(chart_cols,7,figsize=(15,15))

    ax = ax.reshape(-1)

    for i,col in enumerate(df.columns):
        df.boxplot(column=col,ax=ax[i])

def rm_outliers_dict (dataframe, culling_dict):
    ''' Remove values above specified threshold specified for each column
    Must pass in a dictionary, the keys of which are column names  and the values are the outlier thresholds'''
    outliers_list = []
    for col in culling_dict.keys():
        outlier_indices = list(dataframe[dataframe[col] >= culling_dict[col]].index)
        outliers_list = outliers_list + outlier_indices
        
    unique_set = set(outliers_list)
    outliers_list = list(unique_set)
    
    dataframe = dataframe.drop(labels=outliers_list)        
    
    return dataframe

def rm_outliers_threshold (dataframe, columns, threshold,upper=True,lower=True):
    '''Removes values above and below specified threshold for all columns passed as list
    Byt default, function will remove threshold% off the upper and lower ends of the column. 
    To keep outliers in the upper or lower end, change default upper or lower to False'''
    outliers_list = []
    
    for col in columns:
        lower_thresh = dataframe[col].quantile(threshold)
        upper_thresh= dataframe[col].quantile(1 - threshold)

        upper_indices = list(dataframe[dataframe[col] >= upper_thresh].index)
        lower_indices = list(dataframe[dataframe[col] <= lower_thresh].index)
        
        if upper:
            outliers_list = outliers_list + upper_indices
        if lower:
            outliers_list = outliers_list +lower_indices

    unique_set = set(outliers_list)
    outliers_list = list(unique_set)
    
    dataframe = dataframe.drop(labels=outliers_list) 
    return dataframe

def cull_report (dataframe,columns,base_threshold,df_base):
    '''Returns report frame describing row loss from base frame per increment of threshold'''
    
    report_dict = {}
        
    y = int(base_threshold * 100)
    
    for x in range (100,y,-1):
        
        x = float(x/100)
        
        df = rm_outliers_threshold(dataframe,columns,x)
        loss_metrics = row_loss(df_base,df)
        report_dict[str(x)] = loss_metrics
    
    report_df = pd.DataFrame.from_dict(report_dict,orient='index',columns=['rows_dropped','rows_left','row_loss_perc'])
    
    return report_df

def logarize (dataframe,columns):
    
    df = dataframe
    
    for col in columns:
        df[col] =df[col].map(lambda x: np.log(x))
    
    return df

def colinearity_plot(corr,figsize=(12,12)):

    '''non-redundant heatmap of colinearity among columns of dataframe passed in'''
    fig, ax = plt.subplots(figsize=figsize)

    mask = np.zeros_like(corr, dtype=np.bool)
    idx = np.triu_indices_from(mask)
    mask[idx] = True

    sns.heatmap(np.abs(corr),square=True,mask=mask,annot=True,cmap="Reds",ax=ax)
    return fig, ax

def high_corr(dataframe,threshold):
    '''returns multi indexed series of feature pairs with correlation above specified threshold'''
    corr = dataframe.corr()
    sign_corr = corr[abs(corr) > threshold]
    sign_corr = sign_corr.stack()
    sign_corr.drop_duplicates(inplace=True)
    sign_corr = sign_corr[sign_corr != 1]


    return sign_corr

def category_frame (dataframe,categ_cols):
    ''' Uses one-hot encoding on the columns specified in categ_cols and appends them to the parent df '''
    for col in categ_cols:
        
        cat_frame = pd.get_dummies(dataframe[col],drop_first=True)
        cat_frame = cat_frame.astype('int64')
        
        dataframe.drop(col,axis=1,inplace =True)        
        dataframe = dataframe.merge(cat_frame,left_index=True,right_index=True)        
        dataframe.fillna(0)
        
    return dataframe

def min_max_col (series):
    ''' Scales a given series/column values using min max scaling'''
    scaled = (series - min(series)) / (max(series) - min(series))
    return scaled

def df_scaler (dataframe,col_list):
    ''' uses min_max_col functions to scale all columns in the dataframe specified in col_list'''
    for col in col_list:
        dataframe[col] = min_max_col(dataframe[col])
    return dataframe

def pre_process (dataframe,dictionary):
    
    
    dict = dictionary
    
    
    dataframe = rm_outliers_dict(dataframe,dict['categ_culled'])
    
    dataframe = rm_outliers_threshold(dataframe,dict['contin_cull'],dict['contin_cull_thresh'])
    
    dataframe = logarize(dataframe,dict['cols_normed'])
    
    dataframe = category_frame(dataframe,dict['categ_cols'])
    
    
    
    dataframe = dataframe.drop(dict['colinear_columns'],axis=1)    
      
    dataframe = dataframe.drop(dict['remove_cols'],axis=1)
    
    
   
    dataframe = df_scaler(dataframe,list(dataframe.columns))
    
    dataframe = dataframe.fillna(0)
    
    
    print(loss_report(df1,dataframe)) 
    
        
    return dataframe

def remove_pvals (model,dataframe):
    ''' Removes columns representing features with high p-values'''
    pvalues = round(model.pvalues,4)
    pvalues = pvalues.drop('const')
    high_pvalues = pvalues[pvalues > 0.05]    
    high_list = list(high_pvalues.index)

    dataframe = dataframe.drop(high_list,axis=1)
    
    return dataframe 

def test_size_validation (predictors,target):
    collection = []
    size = []
    x= 0.05
    
    while x < 0.95:
        
        errorlist = []
        
        for j in range(1,50):
            
            x_train,x_test,y_train,y_test= train_test_split(predictors,target,test_size=x)
            
            x_train_int= sm.add_constant(x_train)
            
            x_test_int= sm.add_constant(x_test)
            
            olsmod = sm.OLS(y_train,x_train_int).fit()
                       
            y_train_hat = olsmod.predict(x_train_int)
            
            y_test_hat = olsmod.predict(x_test_int)
            
            train_mse = np.sum((y_train - y_train_hat)**2/len(y_train))

            test_mse = np.sum((y_test - y_test_hat)**2/len(y_test))
            
            errorlist.append([train_mse,test_mse])


        saveframe = pd.DataFrame(errorlist,columns=['train','test'])   
        collection.append([str(x), round(saveframe['train'].mean(),3),round(saveframe['test'].mean(),3),0])   

        x = round((x + 0.05),2)

    coll_frame = pd.DataFrame(collection,columns=['size','train','test','delta%'])
    coll_frame['delta%'] = ((coll_frame['test'] - coll_frame['train'])/coll_frame['train']) * 100
    coll_frame['delta%'] = round(coll_frame['delta%'],2)
    coll_frame.set_index('size',inplace=True)    
    
    return coll_frame

def chart_train_test (predictors,target):
    coll_frame = test_size_validation(predictors, target)

    fig, ax = plt.subplots(2,1)

    coll_frame.sort_index(ascending=True,inplace=True)

    ax[0].scatter(coll_frame.index.values,coll_frame['train'],c='red')
    ax[0].scatter(coll_frame.index.values,coll_frame['test'],c='blue')

    coll_frame.sort_values('delta%',ascending=True,inplace=True)
    ax[1].bar(coll_frame.index.values,coll_frame['delta%'])

def mse_validation (predictors,target,size):
    collection = []
              
    errorlist = []
    
    for j in range(1,25):

        x_train,x_test,y_train,y_test= train_test_split(predictors,target,test_size=size)

        x_train_int= sm.add_constant(x_train)

        x_test_int= sm.add_constant(x_test)

        olsmod = sm.OLS(y_train,x_train_int).fit()

        y_train_hat = olsmod.predict(x_train_int)

        y_test_hat = olsmod.predict(x_test_int)

        train_mse = np.sum((y_train - y_train_hat)**2/len(y_train))

        test_mse = np.sum((y_test - y_test_hat)**2/len(y_test))
        
        train_r2 = olsmod.rsquared
        
        train_rmse = sqrt(train_mse)
        
        test_rmse = sqrt(test_mse)
        
        dfx = pd.concat([y_train,y_train_hat],axis =1)
        dfx = dfx[dfx['price'] != 0]
        dfx['diff'] = abs(dfx['price'] - dfx[0])
        dfx = dfx[dfx['diff'] != 0]
        dfx['perc'] = (dfx['diff']/dfx['price'])*100
        train_mape = dfx['perc'].mean()
        
        dfxhat = pd.concat([y_test,y_test_hat],axis =1)
        dfxhat = dfxhat[dfxhat['price'] != 0]
        dfxhat['diff'] = abs(dfxhat['price'] - dfxhat[0])
        dfxhat = dfxhat[dfxhat['diff'] != 0]
        dfxhat['perc'] = (dfxhat['diff']/dfxhat['price'])*100
        test_mape = dfxhat['perc'].mean()
        
        
        errorlist.append([train_mse,test_mse,train_r2,train_rmse,test_rmse,train_mape,test_mape])

    saveframe = pd.DataFrame(errorlist,columns=['train','test','r2','train_rmse','test_rmse','train_mape','test_mape'])
    
    saveframe.fillna(0.0)
    
    report_dict = {}
    
    report_dict['train_mean_squared_error'] = saveframe['train'].mean()
    report_dict['test_mean_squared_error'] = saveframe['test'].mean()
    report_dict['train_rmse'] = saveframe['train_rmse'].mean()
    report_dict['test_rmse'] = saveframe['test_rmse'].mean()  
    report_dict['train_mape'] = saveframe['train_mape'].mean()
    report_dict['test_mape'] = saveframe['test_mape'].mean()
    report_dict['mean_r2'] = round(saveframe['r2'].mean(),2)
    
    
    report_frame = pd.DataFrame.from_dict(report_dict,orient='index',columns=['Scores'])  
    
    return report_frame

def multi_reg_model(dataframe,dictionary,target,rem_pvals = False, test_size = 0.2):
    
    df = pre_process(dataframe,dictionary)
    df = df[df[target] != 0]   
    
    target_series = df[target]
    predictors = df.drop(target,axis=1)
    pred_int = sm.add_constant(predictors)
    
    model = sm.OLS(target_series,pred_int).fit()
    
    if rem_pvals == True:
        df = remove_pvals (model,df)
        
        target_series = df[target]
        predictors = df.drop(target,axis=1)
        
        pred_int = sm.add_constant(predictors)
        model = sm.OLS(target_series,pred_int).fit()
        
    report = mse_validation (predictors,target_series,test_size)
    
    display(model.summary())
    display(report)
    
    return df,model,report