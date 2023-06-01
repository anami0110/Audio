import requests

def obtener_ubicacion_gps():
    # Reemplaza 'TU_CLAVE_DE_API' con tu clave de API de IPStack
    url = 'http://api.ipstack.com/check?access_key=848b5c2c80ae108633703196eb8163f3'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        IP = data['ip']
        latitude = data['latitude']
        longitude = data['longitude']
        country = data['country_name']
        city = data['city']
        region = data['region_name']
        
        
        print(f'Ubicación GPS (IP: {IP}): \n País: {country} \n Ciudad: {city}\n Region: {region} \n Latitud: {latitude}\n Longitud: {longitude}')
        
    except requests.exceptions.RequestException as e:
        print('Error al conectarse al servicio de geolocalización.')

obtener_ubicacion_gps()
