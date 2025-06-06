# SubCSeT 
### (Screening Tools for SUBsurface CO2 storage in petroleum reservoirs on the NCS)  
![](./assets/app_view.png)
The repository contains codes, tools, and data for screening CO2 storage potential of petroleum reservoirs on the Norwegian Continental Shelf (NCS) developed in [WP6 of the CSSR](https://cssr.no/research/fa3/wp-6/), namely:
1.  an open database [*data/_main.csv*](https://github.com/cssr-tools/SubCSeT/blob/main/data/_main.csv) of 134 fields on the NCS compiled from:
    * [FactPages of the Norwegian Offshore Directorate](https://factpages.sodir.no/)  
    * [Public portal of DISKOS database](https://www.diskos.com/) 
    * more references are to be found in the main.ipynb  
2. [*main.ipynb*](https://github.com/cssr-tools/SubCSeT/blob/main/main.ipynb) details the data retrieval, processing, and feature engineering workflows  
3. **web app** to visualize the data and perform screening deployed at **https://subcset-35e143428f88.herokuapp.com/**  

### Publications:  
1. ["GHGT17"](https://github.com/cssr-tools/SubCSeT/releases/tag/GHGT17): the data and the codes presented at the GHGT-17 conference. 
The [**conference paper**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5069175) and [**e-poster**](https://api.ltb.io/show/ACGES) are available at the highlighted links.  

2. "[preprint_IJGGC_special_issue](https://github.com/cssr-tools/SubCSeT/releases/tag/preprint_IJGGC_special_issue)" features the data, codes as of 31.05.2025, prior to submission to the special issue of the International Journal of Greenhouse Gas Control.
The preprint is to be found [here](https://github.com/cssr-tools/SubCSeT/blob/main/preprint_IJGGC_special_issue.pdf).

## Requirements
The **requirements** are listed in *requirements.txt* (for **pip**) and *requirements.yml* (for **conda**). The environment can be reproduced by:  
```
pip install requirements.txt
```  
or 
```
conda env create -n ENV_NAME --file requirements.yml
``` 
or 
```
mamba env create -n ENV_NAME --file requirements.yml
``` 

NB! For Anaconda users. As Anaconda has recently updated its terms of service (read [HERE](https://www.anaconda.com/blog/is-conda-free) and [HERE](https://www.anaconda.com/pricing/terms-of-service-faqs)). So, the `defaults` channel was removed from *requirements.yml*, as an active Anaconda subscription may be needed to use it. 
