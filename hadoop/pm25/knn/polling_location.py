import requests
from bs4 import BeautifulSoup

url = "http://taqm.epa.gov.tw/pm25/tw/PM25A.aspx?area="
get_params = ["1","3","4","6","7","8","9","0"]
get_location = ["北部","竹苗","中部","雲嘉南","高屏","宜蘭","花東","離島"]

all_data = {}

for index, get_index in enumerate(get_params):
    response = requests.get(url+get_index)
    soup = BeautifulSoup(response.text, 'lxml')
    stations_data = soup.find('table','TABLE_G').find_all('tr')
    stations = stations_data[1:]
    i = 0
    for station in stations:
        station_name = station.find('a').getText().strip()
        all_data.update({station_name:get_location[index]})
        i += 1
    
    print (get_location[index] +" "+ str(i))


with open('knn3.txt','r') as file:
    for line in file:
        get_str = line.strip("\r\n")
        knn_station_name = get_str.split("\t")[0]
        n_station_str = get_str.split("\t")[1]
        n_stations = n_station_str.split(" ")
        result = ""
        for n_station in n_stations:
            result += " " + all_data[n_station.split(",")[0]]
       
        print (get_str + " |" +all_data[knn_station_name]+":" + result)
