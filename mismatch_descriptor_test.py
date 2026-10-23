import pandas as pd, numpy as np, joblib, os, json, time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

base=r'D:\wrapper\LFS'
df=pd.read_excel(os.path.join(base,'features_with_predictions-3754.xlsx'))
X=np.zeros((len(df),32)); dt=df['dataset_type'].to_numpy()
for i in range(32):
    tr=df[f'train_feature_{i}'].to_numpy(dtype=float)
    te=df[f'test_feature_{i}'].to_numpy(dtype=float)
    X[:,i]=np.where(dt=='train',tr,te)
y=df['LFS'].to_numpy(dtype=float)
smiles=df['smiles'].to_numpy()
m=joblib.load(os.path.join(base,'重新做不含RP-3的-3754条数据','voting_regressor_model-LFS-3754.pkl'))

# 每个SMILES的代表分子特征: 嵌入(0:16) + Mordred(19:32), 用该燃料任一train行
fuel_mol={}
for smi,sub in df[df['dataset_type']=='train'].groupby('smiles'):
    idx=sub.index[0]
    fuel_mol[smi]=(X[idx,0:16].copy(), X[idx,19:32].copy())

# 审稿人点名的4对: (目标燃料, 错误燃料)
pairs=[('[HH]','CC'),('C','CCCCCCCCCCCCCCCC'),('CC(C)CC(C)(C)C','CCC1CCCCC1'),('CCO','CCCO')]
pnames={'[HH]':'H2','C':'甲烷','CC(C)CC(C)(C)C':'异辛烷','CCO':'乙醇',
        'CC':'乙烷','CCCCCCCCCCCCCCCC':'正十六烷','CCC1CCCCC1':'乙基环己烷','CCCO':'丙醇'}

def build_mismatch(X_row, wrong_mol):
    x=X_row.copy()
    x[0:16]=wrong_mol[0]   # 嵌入换错
    x[19:32]=wrong_mol[1]  # Mordred换错
    return x

results=[]
# 基线: 用主模型预测全部(正确分子信息)
base_pred=m.predict(X)
# 对每个目标燃料, 正常预测与错配预测对比
for tgt, wrong in pairs:
    idx=np.where(smiles==tgt)[0]
    yt=y[idx]; Xt=X[idx]
    pred_norm=base_pred[idx]  # 注意: 主模型见过该燃料train数据, 但对本测试用全量合理
    Xw=np.array([build_mismatch(xi,fuel_mol[wrong]) for xi in Xt])
    pred_wrong=m.predict(Xw)
    results.append({
        'Target':pnames[tgt],'WrongFuel':pnames[wrong],
        'n':int(len(idx)),
        '正常MAE':round(float(mean_absolute_error(yt,pred_norm)),2),
        '错配MAE':round(float(mean_absolute_error(yt,pred_wrong)),2),
        '正常RMSE':round(float(np.sqrt(mean_squared_error(yt,pred_norm))),2),
        '错配RMSE':round(float(np.sqrt(mean_squared_error(yt,pred_wrong))),2),
        '错配MAE/正常MAE':round(float(mean_absolute_error(yt,pred_wrong)/max(mean_absolute_error(yt,pred_norm),1e-9)),2),
        '正常R2':round(float(r2_score(yt,pred_norm)),3),
        '错配R2':round(float(r2_score(yt,pred_wrong)),3),
    })
    print(json.dumps(results[-1],ensure_ascii=False,default=str),flush=True)

# 也做'全局错配'统计: 对每个样本随机换成另一种燃料的分子特征
rng=np.random.RandomState(42)
allfuels=list(fuel_mol.keys())
Xg=X.copy(); masked=0
for i in range(len(X)):
    cur=smiles[i]
    others=[f for f in allfuels if f!=cur]
    w=rng.choice(others)
    Xg[i,0:16]=fuel_mol[w][0]; Xg[i,19:32]=fuel_mol[w][1]
pred_global=m.predict(Xg)
results.append({'Target':'全部样本(随机错配分子)','WrongFuel':'随机其他燃料',
 'n':int(len(X)),'正常MAE':round(float(mean_absolute_error(y,base_pred)),2),
 '错配MAE':round(float(mean_absolute_error(y,pred_global)),2),
 '正常RMSE':round(float(np.sqrt(mean_squared_error(y,base_pred))),2),
 '错配RMSE':round(float(np.sqrt(mean_squared_error(y,pred_global))),2),
 '错配MAE/正常MAE':round(float(mean_absolute_error(y,pred_global)/max(mean_absolute_error(y,base_pred),1e-9)),2),
 '正常R2':round(float(r2_score(y,base_pred)),3),'错配R2':round(float(r2_score(y,pred_global)),3)})
print(json.dumps(results[-1],ensure_ascii=False,default=str),flush=True)

out=pd.DataFrame(results)
out.to_excel(os.path.join(base,'重新做不含RP-3的-3754条数据','mismatch_test_results.xlsx'),index=False)
print('saved mismatch results')
