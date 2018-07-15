####
from urllib.request import urlretrieve
def load_file_from_web(file_url, file_name):
	'''
	Downloads a file from a URL and stores it on the working directory. Usefull when working on cloud environments.
	'''
	url = file_url
	filename = file_name
	urlretrieve(url, filename)

	
####
import zipfile
import pandas as pd
def load_csv_from_zip(zipped_file_name,csv_name):
	'''
	Extracts CSVs from Zipped files and returns the Pandas DataFrame.
	required_libraries = ['zipfile', 'pandas']
	'''
	archive = zipfile.ZipFile(zipped_file_name, 'r')
	df =  pd.read_csv(archive.open(csv_name))
	return df
