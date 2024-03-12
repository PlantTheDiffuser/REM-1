filename = "Pump.SLDPRT"

with open(filename, "rb") as file:
    hex_list = [ord(c) for c in f'{file.read()}']
    print(hex_list)