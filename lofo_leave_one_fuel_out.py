import pandas as pd, numpy as np, joblib, os, json, time
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

base=r'D:\wrapper\LFS'
df=pd.read_excel(os.path.join(base,'features_with_predictions-3754.xlsx'))
# 构造32维X
X=np.zeros((len(df),32)); dt=df['dataset_type'].to_numpy()
for i in range(32):
    tr=df[f'train_feature_{i}'].to_numpy(dtype=float)
    te=df[f'test_feature_{i}'].to_numpy(dtype=float)
    X[:,i]=np.where(dt=='train',tr,te)
y=df['LFS'].to_numpy(dtype=float)
smiles=df['smiles'].to_numpy()

# 主模型超参
def build_model():
    xgb=XGBRegressor(max_depth=4,n_estimators=250,random_state=42,learning_rate=0.25,min_child_weight=4)
    svr=SVR(kernel='rbf',C=30000,gamma=0.1)
    return VotingRegressor(estimators=[('xgboost',xgb),('svr',svr)],weights=[0.5,0.5])

fuels=np.unique(smiles)
results=[]
t0=time.time()
for fuel in fuels:
    mask=smiles!=fuel
    Xtr,ytr=X[mask],y[mask]
    Xte,yte=X[~mask],y[~mask]
    m=build_model(); m.fit(Xtr,ytr)
    pred=m.predict(Xte)
    r2=r2_score(yte,pred); rmse=np.sqrt(mean_squared_error(yte,pred)); mae=mean_absolute_error(yte,pred)
    mape=np.mean(np.abs((yte-pred)/yte))*100
    results.append({'Fuel':fuel,'SMILES':fuel,'n':int(len(yte)),'R2':float(r2),'RMSE':float(rmse),'MAE':float(mae),'MAPE%':float(mape)})
    print(f'{fuel}: n={len(yte)} R2={r2:.3f} RMSE={rmse:.2f} MAE={mae:.2f} MAPE={mape:.1f}%',flush=True)

res=pd.DataFrame(results).sort_values('MAE',ascending=False)
# 汇总
summary={
 'mean_MAE':float(res['MAE'].mean()),'median_MAE':float(res['MAE'].median()),
 'worst_fuel':str(res.iloc[0]['Fuel']),'worst_MAE':float(res.iloc[0]['MAE']),
 'best_fuel':str(res.iloc[-1]['Fuel']),'best_MAE':float(res.iloc[-1]['MAE']),
 'mean_RMSE':float(res['RMSE'].mean()),'mean_R2':float(res['R2'].mean()),
 'pct_MAE_over5':float((res['MAE']>5).mean()*100),
 'pct_MAE_over10':float((res['MAE']>10).mean()*100),
 'pct_MAPE_over20':float((res['MAPE%']>20).mean()*100),
}
outpath=os.path.join(base,'重新做不含RP-3的-3754条数据','lofo_full_results.xlsx')
res.to_excel(outpath,index=False)
json.dump(summary,open(os.path.join(base,'重新做不含RP-3的-3754条数据','lofo_summary.json'),'w'),ensure_ascii=False,indent=1)
print('SUMMARY',json.dumps(summary,ensure_ascii=False,default=str))
print('saved',outpath,'elapsed',round(time.time()-t0,1),'s')
