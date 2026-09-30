import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title='ThermoLift-X FINAL', page_icon='🛢️', layout='wide')
FEATURES=['steam_volume','injection_pressure','soak_days','spm','stroke_fraction','vfd_hz','temperature_c','viscosity_cp','well_pressure_bar','pump_fillage','rod_load']

@st.cache_data
def generate_data(seed=26120,n_wells=24,days=480):
    rng=np.random.default_rng(seed); rows=[]
    for w in range(n_wells):
        well=f'BGW-{w+1:02d}'; baseT=rng.uniform(46,49); baseP=rng.uniform(15,24); baseQ=rng.uniform(25,58); visc0=rng.uniform(5200,8500)
        for d in range(days):
            cycle=d//40; age=d%40
            steam=float(np.clip(rng.normal(430+55*np.sin(cycle*.55),60),240,650)); ip=float(np.clip(rng.normal(25.5,2.8),18,34)); soak=float(np.clip(rng.normal(4,.7),2,6)); spm=float(np.clip(rng.normal(5.7,.65),4,7.2)); stroke=float(np.clip(rng.normal(.80,.075),.58,.96)); vfd=float(np.clip(rng.normal(48,5),34,62))
            memory=sum((430+35*np.sin(c*.55))*np.exp(-.36*(cycle-c)) for c in range(max(0,cycle-6),cycle+1))
            temp=float(np.clip(baseT+.019*memory*np.exp(-.018*age)+.0025*steam-.0055*d+rng.normal(0,.55),44,92)); visc=float(visc0*np.exp(-.046*(temp-47))); wp=float(baseP+.012*steam-.09*cycle+rng.normal(0,.45))
            fill=float(np.clip(.91-.000032*visc+.040*stroke-.040*(spm-5.25)**2+.006*(temp-55)+rng.normal(0,.018),.40,.99)); rod=float(98+.0037*visc+8.8*spm+17.5*stroke+rng.normal(0,4))
            q=float(np.clip(baseQ+1.25*(temp-50)+.028*steam+20*(fill-.70)-.035*(spm-5.5)**2+rng.normal(0,1.8),6,105)); sor=float(np.clip(steam/max(q,1),3,18)); energy=float(np.clip(vfd**2*spm/250+.08*visc/1000+1.5*ip,15,180))
            fail=float(1/(1+np.exp(-(0.055*(rod-150)+1.8*(.66-fill)+.23*(spm-5.5)+.018*(60-temp))))); floating=float(1/(1+np.exp(-(0.075*(rod-155)+2.2*(.64-fill)+.30*(spm-5.5))))); unset=float(1/(1+np.exp(-(0.065*(rod-160)+2.0*(.60-fill)+.02*(50-temp)))))
            rows.append([well,d,cycle,age,steam,ip,soak,spm,stroke,vfd,temp,visc,wp,fill,rod,q,sor,energy,fail,floating,unset])
    return pd.DataFrame(rows,columns=['well_id','day','css_cycle','cycle_day','steam_volume','injection_pressure','soak_days','spm','stroke_fraction','vfd_hz','temperature_c','viscosity_cp','well_pressure_bar','pump_fillage','rod_load','oil_rate','sor','energy_index','failure_risk','rod_floating_risk','pump_unsetting_risk'])

@st.cache_resource
def train_models(df):
    cutoff=int(len(df)*.8); tr=df.iloc[:cutoff]; te=df.iloc[cutoff:]; Xtr=tr[FEATURES]; Xte=te[FEATURES]
    pm=HistGradientBoostingRegressor(max_iter=180,learning_rate=.055,max_leaf_nodes=24,random_state=1).fit(Xtr,tr.oil_rate); sm=HistGradientBoostingRegressor(max_iter=180,learning_rate=.055,max_leaf_nodes=24,random_state=2).fit(Xtr,tr.sor); em=HistGradientBoostingRegressor(max_iter=160,learning_rate=.055,max_leaf_nodes=24,random_state=3).fit(Xtr,tr.energy_index); rm=HistGradientBoostingClassifier(max_iter=150,learning_rate=.06,max_leaf_nodes=20,random_state=4).fit(Xtr,(tr.failure_risk>.5).astype(int))
    return pm,sm,em,rm,{'production_r2':r2_score(te.oil_rate,pm.predict(Xte)),'production_mae':mean_absolute_error(te.oil_rate,pm.predict(Xte)),'sor_mae':mean_absolute_error(te.sor,sm.predict(Xte))}

def thermal_memory(wdf,n=7):
    by=wdf.groupby('css_cycle').steam_volume.mean().tail(n)
    if len(by)==0:return 0.
    latest=int(by.index.max()); return float(sum(v*np.exp(-.36*(latest-c)) for c,v in by.items()))

def state(base,wdf):
    return {'temperature':float(base.temperature_c),'viscosity':float(base.viscosity_cp),'pressure':float(base.well_pressure_bar),'fillage':float(base.pump_fillage),'rod':float(base.rod_load),'oil':float(base.oil_rate),'sor':float(base.sor),'energy':float(base.energy_index),'failure':float(base.failure_risk),'floating':float(base.rod_floating_risk),'unsetting':float(base.pump_unsetting_risk),'memory':thermal_memory(wdf)}

def simulate(base,wdf,steam,inj,soak,spm,stroke,vfd):
    mem=thermal_memory(wdf); temp=float(np.clip(base.temperature_c+.0135*(steam-400)+.75*(soak-4)+.004*(inj-25)+.0016*mem-.22,44,100)); visc=float(7200*np.exp(-.045*(temp-47))); fill=float(np.clip(.92-.000030*visc+.042*stroke-.038*(spm-5.25)**2+.004*(temp-55),.35,.99)); rod=float(96+.00365*visc+8.7*spm+17*stroke+.12*(inj-25)); oil=float(np.clip(base.oil_rate+1.45*(temp-base.temperature_c)+.025*(steam-400)+22*(fill-base.pump_fillage)-.04*(spm-5.5)**2,5,115)); sor=float(max(2.8,steam/max(oil,1))); energy=float(max(10,(vfd**2)*spm/255+.075*visc/1000+1.3*inj)); fail=float(1/(1+np.exp(-(0.052*(rod-150)+1.65*(.66-fill)+.22*(spm-5.5)+.015*(55-temp))))); floating=float(1/(1+np.exp(-(0.070*(rod-155)+2.0*(.64-fill)+.28*(spm-5.5))))); unset=float(1/(1+np.exp(-(0.060*(rod-160)+1.8*(.60-fill)+.018*(52-temp))))); safe=rod<=178 and fill>=.56 and fail<=.68 and floating<=.62 and unset<=.62 and 4<=spm<=7.2 and .58<=stroke<=.96 and 18<=inj<=34; score=oil-3.4*sor-.055*energy-30*fail-12*floating-10*unset
    return {'steam':steam,'inj':inj,'soak':soak,'spm':spm,'stroke':stroke,'vfd':vfd,'temp':temp,'viscosity':visc,'fillage':fill,'rod':rod,'oil':oil,'sor':sor,'energy':energy,'failure':fail,'floating':floating,'unsetting':unset,'safe':safe,'score':score}

@st.cache_data
def optimize(base_tuple,mem_tuple):
    base=pd.Series(dict(zip(['temperature_c','viscosity_cp','pump_fillage','rod_load','oil_rate','sor'],base_tuple))); wdf=pd.DataFrame(mem_tuple,columns=['css_cycle','steam_volume']); rng=np.random.default_rng(26120); out=[]
    for _ in range(7000): out.append(simulate(base,wdf,rng.uniform(280,620),rng.uniform(19,33),rng.uniform(2,6),rng.uniform(4,7.2),rng.uniform(.58,.96),rng.uniform(35,62)))
    for steam in [300,350,400,450,500,550,600]:
      for soak in [2,3,4,5,6]:
       for spm in [4.2,4.8,5.4,6,6.6,7.2]:
        for stroke in [.60,.70,.80,.90,.95]: out.append(simulate(base,wdf,steam,25,soak,spm,stroke,float(np.clip(46+2*(spm-5.5),35,62))))
    c=pd.DataFrame(out); safe=c[c.safe]; best=(safe if len(safe) else c).loc[(safe if len(safe) else c).score.idxmax()].to_dict(); return best,c

df=generate_data(); pm,sm,em,rm,metrics=train_models(df)
with st.sidebar:
    st.header('🎛️ Digital Twin'); well=st.selectbox('Well',sorted(df.well_id.unique())); view=st.radio('Module',['Executive Twin','CSS + SRP Optimizer','SRP Intelligence','Future Simulator','Data + Models']); st.divider(); st.caption('Synthetic prototype data only. Engineering limits are demonstration constraints.')
wdf=df[df.well_id==well].copy(); base=wdf.iloc[-1]; s=state(base,wdf)
labels=[('🌡️ Temp',f"{s['temperature']:.1f} °C"),('🧪 Viscosity',f"{s['viscosity']:,.0f} cP"),('🛢️ Oil',f"{s['oil']:.1f}"),('♨️ Thermal Memory',f"{s['memory']:.0f}"),('💧 SOR',f"{s['sor']:.2f}"),('⚙️ Fillage',f"{s['fillage']*100:.1f}%"),('🪢 Rod Load',f"{s['rod']:.1f}"),('⚠️ Failure',f"{s['failure']*100:.1f}%")]
cs=st.columns(8)
for c,(a,b) in zip(cs,labels): c.metric(a,b)

if view=='Executive Twin':
 st.subheader('1. Complete Well-to-Surface Digital Twin'); g=wdf.tail(220); fig=make_subplots(rows=3,cols=2,subplot_titles=('Thermal state','Production + SOR','Pump fillage','Rod load','Failure / floating','Pressure + steam'))
 for y,n,r,c in [(g.temperature_c,'Temp °C',1,1),(g.viscosity_cp/1000,'Viscosity k cP',1,1),(g.oil_rate,'Oil',2,1),(g.sor,'SOR',2,1),(g.pump_fillage*100,'Fillage %',2,2),(g.rod_load,'Rod load',3,1),(g.failure_risk*100,'Failure %',3,2),(g.rod_floating_risk*100,'Floating %',3,2),(g.well_pressure_bar,'Pressure bar',1,2),(g.steam_volume,'Steam',1,2)]: fig.add_trace(go.Scatter(x=g.day,y=y,name=n),row=r,col=c)
 fig.update_layout(height=850,margin=dict(l=15,r=15,t=50,b=15)); st.plotly_chart(fig,use_container_width=True)
 st.dataframe(pd.DataFrame({'Twin layer':['Reservoir','Wellbore','SRP','Thermal Memory','Risk layer'],'State':['Temperature + viscosity',f"{s['pressure']:.1f} bar + heat response",f"Fillage {s['fillage']*100:.1f}% / load {s['rod']:.1f}",f"Index {s['memory']:.0f}",f"Failure {s['failure']*100:.1f}% / floating {s['floating']*100:.1f}%"]}),hide_index=True,use_container_width=True)

elif view=='CSS + SRP Optimizer':
 st.subheader('2. Coupled CSS + SRP Multi-Objective Optimizer'); st.write('Searches counterfactual futures and jointly considers production, SOR, energy, rod loading, floating and pump-unsetting risk.')
 if st.button('🚀 RUN OPTIMIZER',type='primary'):
  bt=(float(base.temperature_c),float(base.viscosity_cp),float(base.pump_fillage),float(base.rod_load),float(base.oil_rate),float(base.sor)); mt=tuple(wdf.groupby('css_cycle').steam_volume.mean().reset_index().tail(7).itertuples(index=False,name=None))
  with st.spinner('Evaluating thousands of candidate futures...'): best,cand=optimize(bt,mt)
  st.session_state['best']=best; st.session_state['cand']=cand
 if 'best' not in st.session_state: st.info('Press RUN OPTIMIZER.');
 else:
  b=st.session_state['best']; cand=st.session_state['cand']; st.success('SAFETY ENVELOPE PASSED' if b['safe'] else 'NO SAFE PLAN — ENGINEER REVIEW')
  cs=st.columns(6)
  for c,l,v in zip(cs,['Steam','Soak','SPM','Stroke','VFD','Pressure'],[b['steam'],b['soak'],b['spm'],b['stroke']*100,b['vfd'],b['inj']]): c.metric(l,f'{v:.1f}')
  st.subheader('Predicted outcome'); cs=st.columns(6)
  for c,l,v in zip(cs,['Oil','SOR','Energy','Rod load','Failure','Floating'],[b['oil'],b['sor'],b['energy'],b['rod'],b['failure']*100,b['floating']*100]): c.metric(l,f'{v:.2f}')
  safe=cand[cand.safe].nlargest(600,'score'); fig=go.Figure(go.Scatter(x=safe.sor,y=safe.oil,mode='markers',marker=dict(size=6,color=safe.failure,colorscale='Turbo',showscale=True,colorbar=dict(title='Failure risk')))); fig.add_trace(go.Scatter(x=[b['sor']],y=[b['oil']],mode='markers',marker=dict(size=18,symbol='star'),name='Selected')); fig.update_layout(height=500,title='Counterfactual future landscape',xaxis_title='SOR',yaxis_title='Oil rate'); st.plotly_chart(fig,use_container_width=True)
  st.subheader('Why this plan?'); st.write(f"• Thermal response predicts {b['temp']:.1f} °C and {b['viscosity']:.0f} cP."); st.write(f"• Pump fillage = {b['fillage']*100:.1f}%, rod load = {b['rod']:.1f}."); st.write(f"• Rod floating risk = {b['floating']*100:.1f}%; pump-unsetting risk = {b['unsetting']*100:.1f}%."); st.write('• Unsafe candidates are filtered before selecting the multi-objective optimum.')

elif view=='SRP Intelligence':
 st.subheader('3. SRP Intelligence & Failure Prevention'); g=wdf.tail(160); fig=go.Figure();
 for y,n in [(g.rod_load,'Rod load'),(g.pump_fillage*100,'Fillage %'),(g.rod_floating_risk*100,'Floating risk %'),(g.pump_unsetting_risk*100,'Unsetting risk %')]: fig.add_trace(go.Scatter(x=g.day,y=y,name=n))
 fig.update_layout(height=500,xaxis_title='Day'); st.plotly_chart(fig,use_container_width=True); cs=st.columns(4); cs[0].metric('Fillage',f"{s['fillage']*100:.1f}%"); cs[1].metric('Rod load',f"{s['rod']:.1f}"); cs[2].metric('Floating',f"{s['floating']*100:.1f}%"); cs[3].metric('Unsetting',f"{s['unsetting']*100:.1f}%"); st.info('The optimizer treats these as coupled constraints/penalties instead of optimizing production alone.')

elif view=='Future Simulator':
 st.subheader('4. 30-Day Counterfactual Future Simulator'); c=st.columns(3); steam=c[0].slider('Steam',250,650,450,25); soak=c[1].slider('Soak days',2.,6.,4.,.5); spm=c[2].slider('SPM',4.,7.2,5.4,.2); stroke=st.slider('Stroke fraction',.58,.96,.80,.02); vfd=st.slider('VFD Hz',35.,62.,48.,1.); inj=st.slider('Injection pressure',18.,34.,25.,1.); r=simulate(base,wdf,steam,inj,soak,spm,stroke,vfd)
 cs=st.columns(6)
 for x,l,v in zip(cs,['Oil','SOR','Energy','Rod','Failure','Temp'],[r['oil'],r['sor'],r['energy'],r['rod'],r['failure']*100,r['temp']]): x.metric(l,f'{v:.2f}')
 days=np.arange(31); fig=make_subplots(rows=1,cols=3,subplot_titles=('Oil','SOR','Temperature')); fig.add_trace(go.Scatter(x=days,y=base.oil_rate+(r['oil']-base.oil_rate)*(1-np.exp(-days/9))),row=1,col=1); fig.add_trace(go.Scatter(x=days,y=base.sor+(r['sor']-base.sor)*(1-np.exp(-days/7))),row=1,col=2); fig.add_trace(go.Scatter(x=days,y=base.temperature_c+(r['temp']-base.temperature_c)*(1-np.exp(-days/10))),row=1,col=3); fig.update_layout(height=400); st.plotly_chart(fig,use_container_width=True); st.success('SAFE candidate' if r['safe'] else 'OUTSIDE prototype safety envelope')

else:
 st.subheader('5. Data + Models + Validation'); cs=st.columns(4); cs[0].metric('Records',f'{len(df):,}'); cs[1].metric('Wells',df.well_id.nunique()); cs[2].metric('Production R²',f"{metrics['production_r2']:.3f}"); cs[3].metric('Production MAE',f"{metrics['production_mae']:.2f}")
 st.dataframe(pd.DataFrame({'Layer':['Data foundation','Thermal Memory','Production','SOR','Energy','Failure','SRP risk','Optimizer'],'Prototype':['Synthetic operational history','Cycle-decay heat memory','Gradient-boosted regression','Gradient-boosted regression','Gradient-boosted regression','Gradient-boosted classifier','Physics-inspired risk functions','Counterfactual constrained search']}),hide_index=True,use_container_width=True); st.warning('All current values are synthetic. Validation metrics demonstrate software functionality only, not field accuracy.')

st.divider(); st.caption('ThermoLift-X FINAL v3.0 • Human-in-the-loop prototype • Synthetic data only')
