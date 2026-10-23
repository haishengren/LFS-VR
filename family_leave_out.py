import pandas as pd, numpy as np, joblib, os, json, time
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

base=r'D:\wrapper\LFS'
df=pd.read_excel(os.path.join(base,'features_with_predictions-3754.xlsx'))
X=np.zeros((len(df),32)); dt=df['dataset_type'].to_numpy()
for i in range(32):
    tr=df[f'train_feature_{i}'].to_numpy(dtype=float); te=df[f'test_feature_{i}'].to_numpy(dtype=float)
    X[:,i]=np.where(dt=='train',tr,te)
y=df['LFS'].to_numpy(dtype=float); smiles=df['smiles'].to_numpy()

# 分族
alkanes=['C','CC','CCC','CCCC','CCCCC','CCCCCCC','CCCCCCCC','CCCCCCCCC','CCCCCCCCCC','CCCCCCCCCCCC','CCCCCCCCCCCCCC','CCCCCCCCCCCCCCCC','CC(C)CC(C)(C)C','CC(C)CCC(C)C','CCCCC(C)CC','CCCCCCCCCCCCCC(C)C']
alkenes=['C=C','CC=C','CCC=C','C=CC=C']
alcohols=['CO','CCO','CCCO','CCCCO','CCCCCO']
cyclo=['C1CCCCC1','CC1CCCCC1','CCC1CCCCC1','CCCC1CCCCC1','CCCCC1CCCCC1']
aroms=['C1=CC=CC=C1','CC1=CC=CC=C1','CC1=CC=CC=C1C','CC1=CC(=CC=C1)C','CC1=CC(=CC(=C1)C)C','CCC1=CC=CC=C1','CCCC1=CC=CC=C1','CCCCC1=CC=CC=C1']
h2=['[HH]']
groups={'Alkanes(烷烃)':alkanes,'Alkenes(烯烃)':alkenes,'Alcohols(醇类)':alcohols,'Cycloalkanes(环烷烃)':cyclo,'Aromatics(芳烃)':aroms,'H2':h2}

def build_model():
    xgb=XGBRegressor(max_depth=4,n_estimators=250,random_state=42,learning_rate=0.25,min_child_weight=4)
    svr=SVR(kernel='rbf',C=30000,gamma=0.1)
    return VotingRegressor(estimators=[('xgboost',xgb),('svr',svr)],weights=[0.5,0.5])

results=[]; t0=time.time()
for gname, fuels in groups.items():
    held=set(fuels)
    mask=np.isin(smiles,list(held))
    Xtr,ytr=X[~mask],y[~mask]; Xte,yte=X[mask],y[mask]
    m=build_model(); m.fit(Xtr,ytr)
    pred=m.predict(Xte)
    r2=r2_score(yte,pred); rmse=np.sqrt(mean_squared_error(yte,pred)); mae=mean_absolute_error(yte,pred)
    mape=np.mean(np.abs((yte-pred)/yte))*100
    results.append({'族':gname,'燃料数':len(fuels),'样本数':int(len(yte)),'R2':round(float(r2),3),'RMSE':round(float(rmse),2),'MAE':round(float(mae),2),'MAPE%':round(float(mape),1)})
    print(f'{gname}: fuels={len(fuels)} n={len(yte)} R2={r2:.3f} RMSE={rmse:.2f} MAE={mae:.2f} MAPE={mape:.1f}%',flush=True)
res=pd.DataFrame(results)
res.to_excel(os.path.join(base,'重新做不含RP-3的-3754条数据','family_leaveout_results.xlsx'),index=False)
print('saved, elapsed',round(time.time()-t0,1),'s')
