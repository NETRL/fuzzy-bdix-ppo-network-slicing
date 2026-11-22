from pathlib import Path
import ctypes as ct
import functools,hashlib,json,os,shutil,subprocess,sys,tempfile
import numpy as np
import gymnasium as gym
from gymnasium import spaces

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'components'))
SLICES=('eMBB','uRLLC','mMTC')
FEATURES=('backlog_per_1000','oldest_delay_ratio','last_served_delay_ratio',
 'timely_reliability_ratio','offered_rate_per_network_capacity','requested_latency_per_train_p95',
 'requested_reliability','allocated_fraction','isolation_index','recorded_fault_flag','plan_reject','plan_defer','plan_optimize','plan_immediate')
PACKET_COLUMNS=('packet_id','cell','slice_id','created_tick','deadline_tick','departed_tick')
PRB_BANDWIDTH_HZ=360000
SPECTRAL_EFFICIENCY=3.0
DELTA_PRB=5
N_MIN=2
RESIDUAL=0.3
U_GUARD=np.asarray([0.80,0.35,0.80])
OMEGA=np.asarray([0.3,0.5,0.2])
COMMIT_RULE={'eMBB':('median',0.9),'uRLLC':('p95',1.0),'mMTC':('p95',1.0)}
PRB_RATE_MBPS=PRB_BANDWIDTH_HZ*SPECTRAL_EFFICIENCY/1e6
PACKET_BITS=8000
TICK_SECONDS=.001
CONTROL_TICKS=10
FLOORS=np.asarray([30.,15.,1.])

def decode_joint_action(action):
    a=int(action)
    if not 0<=a<64:raise ValueError('DQN joint action must be in 0..63')
    return np.asarray([(a//16)%4,(a//4)%4,a%4],dtype=np.int64)

def project_integer(proposals,budget=100,util=None):

    x=np.maximum(np.rint(np.asarray(proposals,dtype=float)),N_MIN).astype(np.int64)
    b=int(budget)
    if x.shape!=(3,) or b<3*N_MIN:raise ValueError('shape(3,) proposals and a budget of at least 3*N_MIN are required')
    leak=np.zeros(3,dtype=np.int64)
    u=np.zeros(3) if util is None else np.asarray(util,dtype=float)
    order=sorted(range(3),key=lambda i:(u[i]/U_GUARD[i],OMEGA[i]))
    excess=int(x.sum()-b)
    for i in order:
        if excess<=0:break
        cut=min(excess,int(x[i]-N_MIN))
        if cut>0:x[i]-=cut;leak[i]+=cut;excess-=cut
    if x.sum()>b or np.any(x<N_MIN):raise AssertionError('allocator invariant failed')
    return x.astype(np.int32),leak

@functools.lru_cache(maxsize=2)
def load_trace(path=None):
    import pandas as pd
    if path is None:
        path = ROOT / 'data' / 'slicing_dataset_pre_rl.xlsx'
    path=Path(path).resolve();df=pd.read_excel(path)
    values=[];meta={'input':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'rows':len(df),'slice_order':list(SLICES),'train_rows_per_slice':12000,'test_rows_per_slice':3000,
      'split':'Original row order within each slice; first 12,000 for training and last 3,000 held out.',
      'workload_timing':'10 ms demand-holding intervals and 1 ms service ticks; original workbook has no timestamps and its Control_Interval field is not replayed as physical time.',
      'numeric_imputation':'Per-slice training median only; finite positive requested rates/deadlines required.',
      'columns_used':['Requested_DL (Mbps)','Requested_Latency (ms)','Reliability (%)','Fault_Condition','Isolation_Level','Network_Load (%)'],
      'not_used_as_measured_outcomes':['Achieved_DL (Mbps)','Achieved_Latency (ms)','SLA_Met','Reward'],
      'medians':{},'missing_counts':{}}
    for s in SLICES:
        z=df[df.Slice_Type==s].reset_index(drop=True)
        if len(z)!=15000:raise ValueError('Expected15000 records per slice')
        cols=[];meds={};miss={}
        for name in ['Requested_DL (Mbps)','Requested_Latency (ms)','Reliability (%)']:
            a=np.array(pd.to_numeric(z[name],errors='coerce').to_numpy(float),copy=True);valid=np.isfinite(a)&(a>0)
            median=float(np.median(a[:12000][valid[:12000]]))
            meds[name]=median;miss[name]=int((~valid).sum());a[~valid]=median
            cols.append(a)

        fault=z['Fault_Condition'].astype(str).str.upper().isin(['NODE_DOWN','LINK_FAILURE','SIGNAL_DROP']).to_numpy(float)
        cols.append(fault)
        iso_req=z['Isolation_Level'].astype(str).str.upper().map({'LOW':0.,'MEDIUM':.5,'HIGH':1.}).fillna(.5).to_numpy(float)
        cols.append(iso_req)
        ld=z['Network_Load (%)'].astype(str).str.strip().str.upper()
        ldn=pd.to_numeric(ld,errors='coerce')/100.
        ldt=ld.map({'LOW':.3,'MEDIUM':.6,'HIGH':.9})
        cols.append(ldn.fillna(ldt).fillna(.6).to_numpy(float))
        values.append(np.stack(cols,axis=1));meta['medians'][s]=meds;meta['missing_counts'][s]=miss
    data=np.stack(values,axis=1)
    p95=np.percentile(data[:12000,:,1],95,axis=0)
    commit=np.zeros(3)
    for i,s in enumerate(SLICES):
        kind,f=COMMIT_RULE[s];d=data[:12000,i,0]
        commit[i]=f*(np.median(d) if kind=='median' else np.percentile(d,95))
    meta['committed_rate_mbps']={s:float(commit[i]) for i,s in enumerate(SLICES)}
    meta['committed_rate_rule']='eMBB: 0.9 x median requested DL (the SLA proxy floor); uRLLC and mMTC: 95th percentile of the requested DL; the remainder of a request is best effort and is not offered to the slice queue.'
    return data,p95,meta,commit

@functools.lru_cache(maxsize=1)
def backend():
    src=HERE/'trace_engine.cpp';lib=HERE/'libtrace_engine.so'
    if not lib.exists() or lib.stat().st_mtime<src.stat().st_mtime:
        compiler=shutil.which('g++')
        if compiler is None:raise RuntimeError('A C++17 compiler (g++) is required.')
        fd,tmp=tempfile.mkstemp(prefix='trace-engine-',suffix='.so',dir=HERE);os.close(fd)
        try:
            subprocess.run([compiler,'-std=c++17','-O3','-shared','-fPIC',str(src),'-o',tmp],check=True,capture_output=True)
            os.replace(tmp,lib)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    c=ct.CDLL(str(lib));c.tr_create.argtypes=[ct.c_double];c.tr_create.restype=ct.c_void_p
    for n in ['tr_destroy','tr_reset']:getattr(c,n).argtypes=[ct.c_void_p]
    c.tr_step.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int),ct.POINTER(ct.c_double),ct.POINTER(ct.c_double)]
    for n in ['tr_metrics','tr_packets']:getattr(c,n).argtypes=[ct.c_void_p,ct.POINTER(ct.c_double)]
    for n in ['tr_count','tr_tick']:getattr(c,n).argtypes=[ct.c_void_p];getattr(c,n).restype=ct.c_int
    return c

def _reward_functions():
    try:from fuzzy_system import fuzzy_reward,crisp_reward
    except ImportError as e:raise ImportError('The validation components/fuzzy_system.py module is required.') from e
    return fuzzy_reward,crisp_reward

class TraceResourceEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,reward_kind='fuzzy',isolation_enabled=True,retain_records=False,horizon=2000,data_path=None,**kwargs):
        super().__init__()
        if kwargs:raise TypeError('Unknown environment arguments: '+str(list(kwargs)))
        if reward_kind not in ['fuzzy','crisp']:raise ValueError(reward_kind)
        self.data,self.latency_scale,self.preprocessing,self.commit=load_trace(data_path)
        self.reward_kind=reward_kind;self.isolation_enabled=bool(isolation_enabled);self.retain_records=bool(retain_records)
        self.horizon=int(horizon)
        if self.horizon<1 or self.horizon>3000:raise ValueError('horizon must be1..3000')
        self.action_space=spaces.MultiDiscrete([4,4,4]);self.observation_space=spaces.Box(0,np.inf,(42,),np.float32)
        self.lib=backend();self.handle=self.lib.tr_create(PRB_RATE_MBPS*1000)
        self.control_records=[];self._is_reset=False
    def _raw(self):
        if not self.handle:raise RuntimeError('Environment is closed')
        a=np.empty((3,3,10),dtype=np.float64);self.lib.tr_metrics(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_double)));return a
    def _current_row(self):


        i=min(self.start_index+self.steps,self.start_index+self.horizon-1)
        row=self.data[i].copy();row[:,0]*=self.traffic_scale;row[:,1]*=self.sla_scale
        return row
    def _budget(self,step):
        factor=1.
        for start,end,f in self.fault_windows:
            if start<=step<end:factor=min(factor,float(f))
        return int(np.floor(self.budget_prbs*factor))
    def reset(self,seed=None,options=None):
        if not self.handle:raise RuntimeError('Environment is closed')
        super().reset(seed=seed);o=dict(options or {});split=o.pop('split','train')
        if split not in ['train','test','eval']:raise ValueError('split must be train or test')
        low,high=(0,12000) if split=='train' else (12000,15000)
        self.start_index=int(o.pop('start_index',self.np_random.integers(low,high-self.horizon+1)))
        if not low<=self.start_index<=high-self.horizon:raise ValueError('episode start crosses its train/test partition')
        self.split=split;self.traffic_scale=float(o.pop('traffic_scale',1.));self.sla_scale=float(o.pop('sla_scale',1.));self.budget_prbs=int(o.pop('budget_prbs',100))
        self.fault_windows=[tuple(x) for x in o.pop('fault_windows',[])]
        if o:raise ValueError('Unknown reset options: '+str(list(o)))
        if not np.isfinite([self.traffic_scale,self.sla_scale]).all() or self.traffic_scale<=0 or self.sla_scale<=0 or self.budget_prbs<3:raise ValueError('finite positive traffic/target scales and at least3PRBs required')
        for start,end,f in self.fault_windows:
            if not(0<=start<end<=self.horizon and 0<=f<=1):raise ValueError('fault windows are[start,end,factor] within the episode')
        self.steps=0;self.lib.tr_reset(self.handle)
        balanced=np.full(3,self.budget_prbs//3,dtype=np.int32);balanced[:self.budget_prbs%3]+=1
        self.allocations=np.tile(balanced,(3,1));self.last_raw=self._raw();self.control_records=[]
        self.control_messages=np.zeros(3,dtype=int);self.slice_returns=np.zeros(3);self.canonical_returns=np.zeros(3)
        self.isolation_sum=np.zeros(3);self.last_latency=np.zeros(3);self.timely_fraction=np.ones(3)
        self.deferred=np.zeros(3);self.rejected=np.zeros(3,dtype=bool);self.offered=np.minimum(self._current_row()[:,0],self.commit);self.last_plans=['OPTIMIZE']*3
        self.last_iso=np.ones(3);self.last_leak=np.zeros((3,3),dtype=np.int64);self.last_util=np.full((3,3),0.5);self.last_effective=np.zeros(3,dtype=int)
        self.capacity_feasible=True;self.queue_conservation=True;self._is_reset=True;self._done=False
        self.last_reward_inputs=self._inputs(self.last_raw,self._current_row())
        return self._observation(),{'plan_inputs':self.plan_inputs(),'configuration':self.configuration()}
    def configuration(self):
        return {'split':self.split,'start_index':self.start_index,'horizon':self.horizon,'traffic_scale':self.traffic_scale,
         'sla_scale':self.sla_scale,'budget_prbs':self.budget_prbs,'fault_windows':self.fault_windows,'reward_kind':self.reward_kind,
         'isolation_enabled':self.isolation_enabled,'control_interval_s':.01,'tick_s':.001,'packet_bits':PACKET_BITS,
         'prb_bandwidth_hz':PRB_BANDWIDTH_HZ,'spectral_efficiency_bits_per_s_hz':SPECTRAL_EFFICIENCY,
         'prb_rate_mbps':PRB_RATE_MBPS,'delta_prb':DELTA_PRB,'n_min':N_MIN,'residual_share':RESIDUAL,'utilisation_guards':U_GUARD.tolist(),'priority_weights':OMEGA.tolist(),'committed_rate_mbps':self.commit.tolist(),
         'constraint_cost':'Unique deadline-eligible misses becoming due in this control interval divided by unique eligible packets becoming due, zero when none. Mean across three slices.',
         'reliability':'Lossless service; achieved reliability is timely-delivered fraction of deadline-eligible packets. Workbook Reliability is target only.',
         'fault_semantics':'Recorded fault flag is a plan input. Explicit fault_windows separately reduce available PRBs; this is an ideal service fault, not detection validation.'}
    def _inputs(self,raw,row):
        grant_rates=self.allocations.sum(axis=0)*PRB_RATE_MBPS
        latency=np.maximum(self.last_latency,raw[:,:,7].max(axis=0))
        lr=latency/np.maximum(row[:,1],1e-9)
        rr=self.timely_fraction/np.maximum(row[:,2]/100,1e-9)
        load=self.offered/np.maximum(grant_rates,1e-9)
        isolation=self.last_iso
        return lr,rr,load,isolation
    def local_observations(self):
        if not self._is_reset:raise RuntimeError('Call reset first')
        raw=self.last_raw;row=self._current_row();lr,rr,load,iso=self._inputs(raw,row)
        plan_onehot=np.zeros((3,4))
        for i,p in enumerate(self.plans()):plan_onehot[i,['REJECT','DEFER','OPTIMIZE','IMMEDIATE'].index(p)]=1.
        a=np.stack([np.log1p(np.minimum(raw[:,:,2].sum(axis=0)/1000,100)),
          np.log1p(np.minimum(raw[:,:,7].max(axis=0)/row[:,1],100)),np.log1p(np.minimum(self.last_latency/row[:,1],100)),
          np.minimum(rr,2),np.log1p(np.minimum(row[:,0]/(self.budget_prbs*3*PRB_RATE_MBPS),20)),
          np.minimum(row[:,1]/self.latency_scale,20),row[:,2]/100,
          self.allocations.mean(axis=0)/self.budget_prbs,iso,row[:,3],plan_onehot[:,0],plan_onehot[:,1],plan_onehot[:,2],plan_onehot[:,3]],axis=1)
        return a.astype(np.float32)
    def _observation(self):return self.local_observations().reshape(-1)
    def plan_inputs(self):


        row=self._current_row()
        a=np.stack([np.clip(row[:,5],0,1),np.minimum(row[:,1]/self.latency_scale,1),np.clip(row[:,2]/100,0,1),row[:,4],row[:,3]],axis=1)
        return a
    _plan_cache={}
    def plans(self):
        key=(self.budget_prbs,self.traffic_scale,self.sla_scale,tuple(self.fault_windows))
        cache_file=HERE/('plan_table_%s.npy'%hashlib.sha256(repr(key).encode()).hexdigest()[:12])
        if key not in TraceResourceEnv._plan_cache and cache_file.exists():TraceResourceEnv._plan_cache[key]=np.load(cache_file)
        if key not in TraceResourceEnv._plan_cache:
            from fuzzy_system import plan_score,select_plan
            keep=(self.steps,self.start_index);table=np.empty((len(self.data),3),dtype='<U9')
            for i in range(len(self.data)):
                self.start_index,self.steps=i,0;a=self.plan_inputs()
                for k in range(3):
                    sc=float(plan_score(a[k,0],a[k,1],a[k,2],a[k,3],bool(a[k,4]>0.5)));table[i,k]=str(select_plan(sc,bool(a[k,4]>0.5)))
            self.steps,self.start_index=keep;TraceResourceEnv._plan_cache[key]=table;np.save(cache_file,table)
        i=min(self.start_index+self.steps,self.start_index+self.horizon-1)
        return list(TraceResourceEnv._plan_cache[key][i])
    def propose(self,actions,plans=None):

        a=np.asarray(actions)
        if a.shape!=(3,) or np.any(a!=a.astype(int)) or np.any((a<0)|(a>3)):raise ValueError('joint action must contain three integer labels0..3')
        a=a.astype(int).copy();plans=self.plans() if plans is None else list(plans)
        p=self.allocations.astype(np.int64).copy();leak=np.zeros((3,3),dtype=np.int64)
        util=self.last_util
        for s in range(3):
            step=DELTA_PRB
            if plans[s]=='REJECT' and a[s] in (1,3):a[s]=0
            if plans[s]=='DEFER':a[s]=0
            if plans[s]=='IMMEDIATE' and a[s]==1:step=2*DELTA_PRB
            if a[s]==1:p[:,s]=np.minimum(p[:,s]+step,self._budget(self.steps))
            elif a[s]==2:p[:,s]=np.maximum(p[:,s]-step,N_MIN)
            elif a[s]==3:
                for c in range(3):
                    donors=[d for d in range(3) if d!=s and p[c,d]-step>=N_MIN and util[c,d]*p[c,d]/max(p[c,d]-step,1)<=U_GUARD[d]]
                    if donors:
                        d=min(donors,key=lambda x:util[c,x]/U_GUARD[x]);p[c,d]-=step;p[c,s]+=step;leak[c,d]+=step
        return np.maximum(p,N_MIN),leak,a
    def step(self,actions):
        if not self._is_reset or self._done:raise RuntimeError('reset is required before stepping or after episode termination')
        plans=self.plans();p,leak,eff=self.propose(actions,plans);budget=self._budget(self.steps)
        grants=np.zeros((3,3),dtype=np.int32)
        for c in range(3):
            g,l=project_integer(p[c],budget,self.last_util[c]);grants[c]=g;leak[c]+=l
        return self.step_allocations(grants,proposals=p,actions=np.asarray(actions,dtype=int),plans=plans,leakage=leak,effective=eff)
    def step_allocations(self,grants,proposals=None,actions=None,plans=None,leakage=None,effective=None):
        if not self._is_reset or self._done:raise RuntimeError('reset required')
        plans=self.plans() if plans is None else list(plans);leakage=np.zeros((3,3),dtype=np.int64) if leakage is None else np.asarray(leakage)
        self.last_effective=np.asarray(actions if effective is None else effective)
        g=np.asarray(grants)
        if g.shape==(3,):g=np.tile(g,(3,1))
        budget=self._budget(self.steps)
        if g.shape!=(3,3) or np.any(g!=g.astype(int)) or np.any(g<0) or np.any(g.sum(axis=1)>budget):raise ValueError('integer grants exceed available per-cell budget')
        g=np.ascontiguousarray(g,dtype=np.int32);row=self._current_row();old=self.last_raw;before_alloc=self.allocations.copy()
        before=self.allocations.astype(np.int64).copy()
        committed=np.minimum(row[:,0],self.commit*self.traffic_scale)
        offered=committed.copy()
        for s in range(3):
            if plans[s] in ('REJECT','DEFER'):offered[s]=RESIDUAL*committed[s]
            self.deferred[s]=0.
        rates=np.ascontiguousarray(offered,dtype=float);lat=np.ascontiguousarray(row[:,1],dtype=float)
        self.rejected=np.array([plans[s]=='REJECT' for s in range(3)]);self.last_plans=plans
        self.offered=offered
        self.lib.tr_step(self.handle,g.ctypes.data_as(ct.POINTER(ct.c_int)),rates.ctypes.data_as(ct.POINTER(ct.c_double)),lat.ctypes.data_as(ct.POINTER(ct.c_double)))
        self.allocations=g;raw=self._raw();delta=raw-old
        sc=delta[:,:,1].sum(axis=0);ds=delta[:,:,3].sum(axis=0)
        self.last_latency=np.divide(ds,sc,out=np.zeros(3),where=sc>0)
        due=delta[:,:,4].sum(axis=0);miss=delta[:,:,5].sum(axis=0)
        self.timely_fraction=np.divide(due-miss,due,out=np.ones(3),where=due>0)
        self.last_leak=leakage;self.last_iso=np.clip(1.-leakage/np.maximum(before,1),0.,1.).mean(axis=0)
        self.last_util=np.minimum((self.offered/3.)[None,:]/np.maximum(g*PRB_RATE_MBPS,1e-9),10.)
        inputs=self._inputs(raw,row);fuzzy,crisp=_reward_functions()
        kw=dict(latency_ratio=inputs[0],reliability_ratio=inputs[1],load_ratio=inputs[2],isolation=inputs[3],slice_type=SLICES)
        canonical=np.asarray(fuzzy(**kw,isolation_enabled=True),dtype=float)
        actual=canonical.copy() if self.reward_kind=='fuzzy' and self.isolation_enabled else np.asarray((fuzzy if self.reward_kind=='fuzzy' else crisp)(**kw,isolation_enabled=self.isolation_enabled),dtype=float)
        changed=g!=before_alloc;messages=changed.sum(axis=0)
        self.control_messages+=messages;self.slice_returns+=actual;self.canonical_returns+=canonical;self.isolation_sum+=inputs[3]
        costs=np.divide(miss,due,out=np.zeros(3),where=due>0)
        self.last_raw=raw;self.last_reward_inputs=inputs;self.steps+=1;self._done=self.steps>=self.horizon
        self.capacity_feasible &= bool(np.all(g.sum(axis=1)<=budget))
        self.queue_conservation &= bool(np.all(raw[:,:,0]==raw[:,:,1]+raw[:,:,2]))
        record={'step':self.steps-1,'trace_index':self.start_index+self.steps-1,'time_ms':self.steps*10,
          'actions':None if actions is None else actions.tolist(),'proposals':g.tolist() if proposals is None else np.asarray(proposals).tolist(),
          'grants':g.tolist(),'available_prbs_per_cell':[budget]*3,'projection_adjustment':(g-(g if proposals is None else proposals)).tolist(),
          'prb_ownership_ranges':np.stack([np.cumsum(g,axis=1)-g,np.cumsum(g,axis=1)],axis=2).tolist(),
          'offered_mbps':rates.tolist(),'requested_latency_ms':lat.tolist(),'requested_reliability_pct':row[:,2].tolist(),
          'arrivals':delta[:,:,0].astype(int).tolist(),'departures':delta[:,:,1].astype(int).tolist(),
          'backlog':raw[:,:,2].astype(int).tolist(),'allocated_capacity_bits':delta[:,:,8].tolist(),
          'served_bits':(delta[:,:,1]*PACKET_BITS).astype(int).tolist(),'slice_rewards':actual.tolist(),
          'canonical_slice_rewards':canonical.tolist(),'constraint_costs':costs.tolist(),'eligible_arrivals':due.astype(int).tolist(),
          'eligible_misses':miss.astype(int).tolist(),'isolation':inputs[3].tolist(),
          'capacity_feasible':self.capacity_feasible,'queue_conservation':self.queue_conservation}
        if self.retain_records:self.control_records.append(record)
        info={**record,'slice_rewards':actual,'canonical_slice_rewards':canonical,'constraint_costs':costs,
          'constraint_cost':float(costs.mean()),'plan_inputs':self.plan_inputs()}
        if self._done:info['episode_summary']=self.episode_summary()
        return self._observation(),float(actual.mean()),False,self._done,info
    def packet_records(self):
        if not self.handle:raise RuntimeError('Environment is closed')
        n=self.lib.tr_count(self.handle);a=np.empty((n,6),dtype=float)
        if n:self.lib.tr_packets(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_double)))
        return a
    def episode_summary(self):
        raw=self.last_raw;packets=self.packet_records();duration=self.steps*.01;rows=[]
        for i,s in enumerate(SLICES):
            z=packets[packets[:,2]==i];served=z[z[:,5]>=0];delay=served[:,5]-served[:,3]
            arrived=int(raw[:,i,0].sum());dep=int(raw[:,i,1].sum());eligible=int(raw[:,i,4].sum());misses=int(raw[:,i,5].sum())
            rows.append({'slice':s,'arrived_packets':arrived,'served_packets':dep,'backlog_packets':int(raw[:,i,2].sum()),
              'mean_latency_ms':float(delay.mean()) if len(delay) else None,'p95_latency_ms':float(np.percentile(delay,95)) if len(delay) else None,
              'p99_latency_ms':float(np.percentile(delay,99)) if len(delay) else None,'aggregate_throughput_mbps':dep*PACKET_BITS/max(duration,1e-9)/1e6,
              'deadline_eligible_arrivals':eligible,'deadline_eligible_misses':misses,'deadline_eligible_miss_pct':100*misses/eligible if eligible else 0.,
              'timely_delivery_reliability':1-misses/eligible if eligible else 1.,'isolation_mean':self.isolation_sum[i]/max(self.steps,1),
              'control_messages_per_s':float(self.control_messages[i]/max(duration,1e-9)),
              'mean_reward':float(self.slice_returns[i]/max(self.steps,1)),'canonical_mean_reward':float(self.canonical_returns[i]/max(self.steps,1)),
              'excluded_recent_arrivals':arrived-eligible})
        capbits=raw[:,:,8].sum(axis=1);servedbits=raw[:,:,1].sum(axis=1)*PACKET_BITS
        return {'rows':rows,'total_reward':float(self.slice_returns.mean()),'slice_returns':self.slice_returns.tolist(),
          'canonical_total_reward':float(self.canonical_returns.mean()),'canonical_slice_returns':self.canonical_returns.tolist(),
          'capacity_feasible':bool(self.capacity_feasible and np.all(servedbits<=capbits+1e-6)),
          'queue_conservation':bool(self.queue_conservation),'duration_s':duration,'configuration':self.configuration(),
          'served_bits_per_cell':servedbits.tolist(),'allocated_capacity_bits_per_cell':capbits.tolist()}
    def export(self,prefix):
        import csv
        p=Path(prefix);p.parent.mkdir(parents=True,exist_ok=True)
        with p.with_suffix('.packets.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(PACKET_COLUMNS);w.writerows(self.packet_records())
        p.with_suffix('.controls.json').write_text(json.dumps(self.control_records,indent=2,allow_nan=False))
        p.with_suffix('.summary.json').write_text(json.dumps(self.episode_summary(),indent=2,allow_nan=False))
    def close(self):
        if getattr(self,'handle',None):self.lib.tr_destroy(self.handle);self.handle=None
        self._is_reset=False;self._done=True
    def __del__(self):self.close()
