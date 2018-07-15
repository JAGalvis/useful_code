def load_file_from_web(file_url, file_name):
	from urllib.request import urlretrieve
	url = file_url
	filename = file_name
	urlretrieve(url, filename)
