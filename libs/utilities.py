import re, os, requests, magic, configparser
from urllib.parse import unquote, urlparse
from bs4 import BeautifulSoup


def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = cd.split('filename=')[1]
    if fname.lower().startswith(("utf-8''", "utf-8'")):
        fname = fname.split("'")[-1]
    return unquote(fname)
  
def download_file(url):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        filename = get_filename_from_cd(r.headers.get('content-disposition'))
        if not filename:
            filename = urlparse(url).geturl().replace('https://', '').replace('/', '-')
        filename = 'content/' + filename
        if os.path.isdir("content") == False:
            os.mkdir("content")
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return filename
      
def readtext(path, filter_id=None):
    """Read a local text file or remote web html and return its text content.
    For HTML files, you can filter on html tag id for best importation
    """
    path = path.rstrip()
    path = path.replace(' \n', '')
    path = path.replace('%0A', '')
    if re.match(r'^https?://', path):
        filename = download_file(path)
    else:
        relative_path = path
        filename = os.path.abspath(relative_path)
    
    filetype = magic.from_file(filename, mime=True)
    print(f"\Loading {filename} as {filetype}")
    text = ""
    if filetype == 'application/pdf':
        print('PDF not supported yet')
    if filetype == 'text/plain':
        with open(filename, 'rb') as f:
            text = f.read().decode('utf-8')
    if filetype == 'text/html':
        with open(filename, 'rb') as f:
            soup = BeautifulSoup(f, 'html.parser')
            if filter_id is not None:
                element = soup.find(attrs={'id':filter_id})
                if element is not None:
                    text = element.get_text()
                else:
                    text = soup.get_text()
            else:
                text = soup.get_text()
    
    if os.path.exists(filename) and filename.find('content/') > -1:
        os.remove(filename)
      
    return text

def getconfig(id_cfg="main"):
    config = configparser.ConfigParser()
    config.read('config.ini')
    return dict(config.items(id_cfg))


def list_docs_parser(list_filename):
    # read a file contains a list of files 
    # return a list of urls or paths strings
    file_list = []
    with open(list_filename, 'r', encoding='utf-8') as f:
        for line in f:
            file_list.append(line.strip())
    
    return file_list