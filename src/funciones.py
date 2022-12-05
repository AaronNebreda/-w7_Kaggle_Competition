import pandas as pd
    
    
from sklearn.linear_model import LinearRegression as LinReg

from sklearn.linear_model import Lasso        # regularizacion L1
from sklearn.linear_model import Ridge        # regularizacion L2
from sklearn.linear_model import ElasticNet   # regularizacion L1+L2


    # se inician los modelos
from sklearn.svm import SVR  # support vector regressor

from sklearn.ensemble import RandomForestRegressor as RFR  
from sklearn.tree import ExtraTreeRegressor as ETR


    #%pip install xgboost

    #%pip install catboost

    #%pip install lightgbm

from sklearn.ensemble import GradientBoostingRegressor as GBR

from xgboost import XGBRegressor as XGBR

from catboost import CatBoostRegressor as CTR

from lightgbm import LGBMRegressor as LGBMR

from sklearn.metrics import mean_squared_error as mse 

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import r2_score as r2



def coach_models(X_train, y_train, X_test, y_test):
    
    
    

    lista = []
    
    linreg=LinReg()
    lasso=Lasso()
    ridge=Ridge()
    elastic=ElasticNet()
    svr=SVR()
    rfr=RFR()
    etr=ETR()
    gbr=GBR()
    xgbr=XGBR()
    ctr=CTR()
    lgbmr=LGBMR()
    
    modelos=[linreg, lasso, ridge, elastic, svr, rfr, etr, gbr, xgbr, ctr,  lgbmr]
    n_modelos=['linreg', 'lasso', 'ridge', 'elastic', 'svr', 'rfr', 'etr', 'gbr', 'xgbr', 'ctr', 'lgbmr']

    for i, m in enumerate(modelos):
        linea = []
        linea.append(n_modelos[i])
    
        if n_modelos[i] == 'ctr':
            m.fit(X_train, y_train, verbose=0)
            m.predict(X_test)
            y_pred=m.predict(X_test)
        
            e1 = mse(y_test, y_pred, squared=False)
            e2 = mae(y_test, y_pred)
            e3 = r2(y_test, y_pred)
        
        else:
            m.fit(X_train, y_train)
            m.predict(X_test)
            y_pred=m.predict(X_test)
            
            e1 = mse(y_test, y_pred, squared=False)
            e2 = mae(y_test, y_pred)
            e3 = r2(y_test, y_pred)
            
        linea.append(e1)
        linea.append(e2)
        linea.append(e3)
        
        lista.append(linea)  
            
    df=pd.DataFrame(lista, columns=['modelo', 'rmse', 'mae', 'r2'])
    
    return df