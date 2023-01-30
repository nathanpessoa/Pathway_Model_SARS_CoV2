# Pathway_Model_SARS_CoV2

First step: Download the file 'time_series_covid19_deaths_global.csv' from 
https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases?force_layout=desktop

Second step: run the file 'reading_death.py' in the folder containing the file 'time_series_covid19_deaths_global.csv' to generate separate files with the total number of deaths due to SARS-CoV2 for each country

Third step: run the file 'dDdt.py' separately for each country to obtain the file with the corresponding daily number of deaths

Fourth step: once you plot the data for the daily number of deaths of a given country and discover the number of COVID-19 waves that occurred up to a given date, run one of the following files: 'fit_pathway_model_2nd_wave_v2.py', 'fit_pathway_model_3rd_wave_v2.py', 'fit_pathway_model_4th_wave_v2.py', 'fit_pathway_model_5th_wave_v2.py', according to the number of waves.

%%%%%%%%%%%%%%%%%%%%%%

Note: in the last two steps, the file must be edited in order to analyse the country of interest

%%%%%%%%%%%%%%%%%%%%%%

For more information about the fitting procedure, visit https://link.springer.com/article/10.1007/s11071-022-08179-8
