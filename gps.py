import requests

def obtener_ubicacion_gps():
    # Reemplaza 'TU_CLAVE_DE_API' con tu clave de API de IPStack
    url = 'http://api.ipstack.com/check?access_key=848b5c2c80ae108633703196eb8163f3'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        latitude = data['latitude']
        longitude = data['longitude']
        
        print(f'Ubicación GPS: {latitude}, {longitude}')
        
    except requests.exceptions.RequestException as e:
        print('Error al conectarse al servicio de geolocalización.')

obtener_ubicacion_gps()
