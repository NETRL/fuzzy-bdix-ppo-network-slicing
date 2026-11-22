#include <vector>
#include <array>
#include <deque>
#include <queue>
#include <cmath>
#include <algorithm>
#include <cstdint>

struct Packet { int id,cell,slice,created,departed; double deadline; };
struct Due {double deadline;int id;bool operator<(const Due&o)const{return deadline>o.deadline||(deadline==o.deadline&&id>o.id);} };
struct Engine {
 int tick=0; double pbits=8000., bits_per_prb_tick=720.;
 std::vector<Packet> packets;
 std::array<std::deque<int>,9> queues;
 std::priority_queue<Due> deadlines;
 std::array<double,9> arrivals_credit{}, service_credit{}, arrived{}, served{}, latency_sum{}, eligible{},misses{},ontime{},capacity_bits{};
 Engine(double bits):bits_per_prb_tick(bits){}
 void reset(){tick=0;packets.clear();deadlines={};for(auto&q:queues)q.clear();arrivals_credit.fill(0);service_credit.fill(0);arrived.fill(0);served.fill(0);latency_sum.fill(0);eligible.fill(0);misses.fill(0);ontime.fill(0);capacity_bits.fill(0);}
 void step(const int* grants,const double* rates,const double* latency){
  for(int sub=0;sub<10;++sub){
   for(int cell=0;cell<3;++cell)for(int s=0;s<3;++s){
    int k=cell*3+s;
    arrivals_credit[k]+=rates[s]*1000./(3.*pbits);
    int n=(int)std::floor(arrivals_credit[k]+1e-12);arrivals_credit[k]-=n;
    for(int j=0;j<n;++j){
     int id=packets.size();Packet p{id,cell,s,tick,-1,tick+latency[s]};
     packets.push_back(p);queues[k].push_back(id);deadlines.push({p.deadline,id});++arrived[k];
    }
    double bits=grants[k]*bits_per_prb_tick;capacity_bits[k]+=bits;
    if(queues[k].empty()){service_credit[k]=0;continue;}
    service_credit[k]+=bits/pbits;
    int ns=std::min((int)std::floor(service_credit[k]+1e-12),(int)queues[k].size());
    for(int j=0;j<ns;++j){int id=queues[k].front();queues[k].pop_front();packets[id].departed=tick+1;++served[k];latency_sum[k]+=tick+1-packets[id].created;}
    service_credit[k]-=ns;
    if(queues[k].empty())service_credit[k]=0;
   }
   ++tick;
   while(!deadlines.empty()&&deadlines.top().deadline<=tick+1e-10){
    auto d=deadlines.top();deadlines.pop();const Packet&p=packets[d.id];int k=p.cell*3+p.slice;++eligible[k];
    if(p.departed>=0&&p.departed<=p.deadline+1e-10)++ontime[k];else ++misses[k];
   }
  }
 }
 void metrics(double*out){
  for(int k=0;k<9;++k){
   int z=k*10;out[z]=arrived[k];out[z+1]=served[k];out[z+2]=queues[k].size();out[z+3]=latency_sum[k];
   out[z+4]=eligible[k];out[z+5]=misses[k];out[z+6]=ontime[k];
   out[z+7]=queues[k].empty()?0:tick-packets[queues[k].front()].created;
   out[z+8]=capacity_bits[k];out[z+9]=service_credit[k];
  }
 }
};
extern "C" {
 void* tr_create(double bits){return new Engine(bits);}
 void tr_destroy(void*p){delete (Engine*)p;}
 void tr_reset(void*p){((Engine*)p)->reset();}
 void tr_step(void*p,const int*g,const double*r,const double*l){((Engine*)p)->step(g,r,l);}
 void tr_metrics(void*p,double*out){((Engine*)p)->metrics(out);}
 int tr_count(void*p){return ((Engine*)p)->packets.size();}
 int tr_tick(void*p){return ((Engine*)p)->tick;}
 void tr_packets(void*p,double*out){
  const auto&v=((Engine*)p)->packets;for(size_t i=0;i<v.size();++i){const Packet&q=v[i];out[i*6]=q.id;out[i*6+1]=q.cell;out[i*6+2]=q.slice;out[i*6+3]=q.created;out[i*6+4]=q.deadline;out[i*6+5]=q.departed;}
 }
}
