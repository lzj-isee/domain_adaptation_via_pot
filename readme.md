### v11  
two fc layers, no normalization, auto_threshold, extra_entropy, auto_b

### v10  
two fc layers, no normalization, auto_threshold, no_entropy
  
### v9  
Semi-Supervised Domain Adaptation via Minimax Entropy + Partial OT  

### v8  
loss as cost matrix for partial OT calculation  

### v5_4  
alpha = 0.2, lambda_t = 0.02

### v5  
try to adopt the "classifier_mme" in DANCE method, and the Neighborhood Clustering technique to further enhance Partial DeepJDOT

### v4_2  
1. try to fix the mistake of network design in v4.  
2. Two FC layers increase the accuracy a lot, but why DANCE could use only one FC layer?  
3. remeber to modify the code of dataloader for open set Office31  
4. network design and param_b are so important

### v4  
DeepJDOT may not be suitable in A2D_CDA setting

### v3  
auto tunning param_b, poor performance in A2D_CDA setting, well performance in A2D_OPDA setting  

### v2  
no auto tunning version of v3, has a mature loss function 

### master  
prototype, simple loss function design