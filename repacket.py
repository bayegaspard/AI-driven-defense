import pandas as pd

# Example CSV record (as provided by you)
record = "192.168.10.5-8.254.250.126-49188-80-6,8.254.250.126,80.0,192.168.10.5,49188.0,6.0,03/07/2017 08:55:58,1.0,2.0,0.0,12.0,0.0,6.0,6.0,6.0,0.0,0.0,0.0,0.0,0.0,12000000.0,2000000.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,40.0,0.0,2000000.0,0.0,6.0,6.0,6.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,9.0,6.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,12.0,0.0,0.0,329.0,-1.0,1.0,20.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,BENIGN"

# Split the CSV record into columns
fields = record.split(',')

# Reconstruct original packet information
packet_info = {
    'Flow ID': fields[0],
    'Destination IP': fields[1],
    'Destination Port': int(float(fields[2])),
    'Source IP': fields[3],
    'Source Port': int(float(fields[4])),
    'Protocol': int(float(fields[5])),
    'Timestamp': fields[6],
    'Label': fields[-1].strip()
}

# Output the reconstructed packet
print("Original Packet Format:")
for key, value in packet_info.items():
    print(f"{key}: {value}")
