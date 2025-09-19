Pattern-based fingerprint analysis using ForceSMIP dataset at monthly timesteps. Work at LLNL.    

 
* ```eof-pr.ipynb```: EOF and PC just for precipitation.  
* ```eofs_all_annual.py```: solving EOFs for annual timesteps; ForceSMIP.  
* ```eofs_cmip5_annual.py```: solving EOFs for annual timesteps; CMIP5. 
To be noted: PCs are not normalized in ```eofs*.py```. Normalization and flips are done in ```obs/```. 



obs:
* ```ToE-stochastic_cmip6_pr.ipynb```: stochastic uncertainty assessment for CMIP6 models.   
* ```ToE-stochastic_cmip6_pr_AllMvsMbyM.ipynb```: stochastic uncertainty assessment for CMIP6 models, but also takes the non-detectable members into consideration.    
* ```pr_all.ipynb```, ```pr_all_land.ipynb```, ```monmaxpr_all.ipynb```, ```monmaxpr_all_land.ipynb```, ```tos_all.ipynb```, ```tos_BASIN.ipynb```: SN, ToE figures. 
* ```get_metrics_DATA.py```: calculate signal, noise, SNR for DATA. 
* ```get_metrics_calculation_CMIP.py```: calculate signal, noise, SNR for CMIP simulations. 
* ```enso-pr.ipynb```: Niño 3.4 with precipitation.  
* ```PC-SN-OceanBasins.ipynb```: PC and SNR with respect to ocean basins.   


ModelSpecific:  
As the name indicates, model-specific fingerprints. 

Manuscript coming soon.   
