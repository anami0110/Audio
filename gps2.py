import requests


def get_ip():
    response = requests.get('https://api64.ipify.org?format=json').json()
    return response["ip"]


def get_location():
    ip_address = get_ip()
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_data = {
        "IP": ip_address,
        "Ciudad": response.get("city"),
        "Región": response.get("region"),
        "País": response.get("country_name")
        "Latitud": response.get("latitude")
        "Longitud": response.get("longitude")
    }
    return location_data


print(get_location())
