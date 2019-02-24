####
def load_file_from_web(file_url, file_name):
    from urllib.request import urlretrieve
	'''
	Downloads a file from a URL and stores it on the working directory. Usefull when working on cloud environments.
	'''
	url = file_url
	filename = file_name
	urlretrieve(url, filename)
    
####
def load_gdrive_on_colab(path = '/content/gdrive', wd = '/My Drive/Colab Notebooks/'):
    '''
        Mounts the google drive on Colab's environment.
        Returns the path to 'wd' folder on drive.
    '''
    from google.colab import drive
    drive.mount(path)
    return path + wd

####
def load_file_from_gcs_on_colab():
    '''
        https://colab.research.google.com/notebooks/io.ipynb#scrollTo=z1_FuDjAozF1
    '''
    # Authenticate to GCS.
    from google.colab import auth
    auth.authenticate_user()

    # Create the service client.
    from googleapiclient.discovery import build
    gcs_service = build('storage', 'v1')

    from apiclient.http import MediaIoBaseDownload

    with open('/tmp/downloaded_from_gcs.txt', 'wb') as f:
        request = gcs_service.objects().get_media(bucket=bucket_name,
                                                object='to_upload.txt')
        media = MediaIoBaseDownload(f, request)
        
        done = False
        while not done:
            # _ is a placeholder for a progress object that we ignore.
            # (Our file is small, so we skip reporting progress.)
            _, done = media.next_chunk()
    print('Download complete')

####
def load_csv_from_zip(zipped_file_name,csv_name):
    import zipfile
    import pandas as pd    
	'''
	Extracts CSVs from Zipped files and returns the Pandas DataFrame.
	required_libraries = ['zipfile', 'pandas']
	'''
	archive = zipfile.ZipFile(zipped_file_name, 'r')
	df =  pd.read_csv(archive.open(csv_name))
	return df

####
def check_cloud():
    '''
    Check if the notebook is being run on Datalab or on a local machine.
    Returns True if Cloud, False if local
    '''
    import os
    is_cloud = 'DATALAB_ENV' in list(os.environ)
    return is_cloud

####
def how_many_cores():
	import multiprocessing
	cores = multiprocessing.cpu_count()
	return cores

####
def files_in_path(path_to_check, extension = ''):
    '''
    Returns the files in the path (excludes files in lower level directories and those directory names).
    If want to include only files with a given extension use, for instance, extension = 'csv'
    '''
    from os import walk    
    _ = walk('processed_data/')
    files = list(_)[0][-1]
    if extension == '':
        return files
    else:
        sel_files = []
        for fl in files:
            if '.'+extension in fl:
                sel_files.append(fl)
        return sel_files
