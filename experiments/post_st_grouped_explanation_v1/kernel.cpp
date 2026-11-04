// Per-company descriptive means over the three overlap strata. Inputs are
// native CIF contributions and centered, unit-length ranks of their magnitudes.
#include <cmath>
#include <cstdint>
#include <vector>
extern "C" void pair_scores(int firms,int dim,int topk,const double* x,const double* rank,
 const int* best,const int* overlap,double* out,double* per_axis) {
  constexpr int models=120, metrics=7;
  for(int firm=0;firm<firms;++firm){
    const auto offset=static_cast<long>(firm)*models*dim;
    const double* values=x+offset;
    const double* ranks=rank+offset;
    const int* order=best+static_cast<long>(firm)*models*topk;
    std::vector<double> magnitude(models);
    std::vector<uint64_t> pos(models),neg(models),top(models);
    for(int i=0;i<models;++i){
      for(int k=0;k<dim;++k){
        const double v=values[i*dim+k];magnitude[i]+=std::abs(v);
        if(dim<=64 && v>=0.001)pos[i]|=(uint64_t(1)<<k);
        if(dim<=64 && v<=-0.001)neg[i]|=(uint64_t(1)<<k);
      }
      if(dim<=64)for(int k=0;k<topk;++k){int idx=order[i*topk+k];if(idx>=0)top[i]|=(uint64_t(1)<<idx);}
    }
    double totals[3][metrics]={}, counts[3][metrics]={};
    // Raw per-axis counts enable conditional agreement and material coverage.
    for(int i=0;i<models;++i)for(int j=i+1;j<models;++j){
      const int s=overlap[i*models+j];double l1=0,dot=0;
      int both=0,agree=0,inter=0,uni=0;
      for(int k=0;k<dim;++k){
        const double a=values[i*dim+k],b=values[j*dim+k];
        l1+=std::abs(a-b);dot+=ranks[i*dim+k]*ranks[j*dim+k];
        if(dim>64 || per_axis){
          bool material=std::abs(a)>=.001 && std::abs(b)>=.001;
          if(material){
            bool same=(a>0)==(b>0);both++;agree+=same;
            if(per_axis){
              auto loc=((static_cast<long>(firm)*3+s)*dim+k)*2;
              per_axis[loc]+=same;per_axis[loc+1]++;
            }
          }
        }
      }
      if(dim<=64){
        both=__builtin_popcountll((pos[i]|neg[i])&(pos[j]|neg[j]));
        agree=__builtin_popcountll((pos[i]&pos[j])|(neg[i]&neg[j]));
        inter=__builtin_popcountll(top[i]&top[j]);uni=__builtin_popcountll(top[i]|top[j]);
      }else{
        for(int a=0;a<topk;++a)if(order[i*topk+a]>=0){
          ++uni;for(int b=0;b<topk;++b)inter+=(order[i*topk+a]==order[j*topk+b]);
        }
        for(int b=0;b<topk;++b)uni+=(order[j*topk+b]>=0);
        uni-=inter;
      }
      auto add=[&](int k,double v){totals[s][k]+=v;counts[s][k]++;};
      if(std::isfinite(dot))add(0,dot);
      if(uni)add(1,double(inter)/uni);
      add(2,l1);
      if(magnitude[i]+magnitude[j]>0)add(3,l1/(magnitude[i]+magnitude[j]));
      if(both)add(4,double(agree)/both);
      add(5,double(both)/dim);
      add(6,both>0);
    }
    for(int s=0;s<3;++s)for(int k=0;k<metrics;++k){
      auto idx=((firm*3+s)*metrics+k)*2;
      out[idx]=counts[s][k]?totals[s][k]/counts[s][k]:NAN;out[idx+1]=counts[s][k];
    }
  }
}
