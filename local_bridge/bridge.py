import serial
import serial.tools.list_ports
import socketio
import time
import sys

# Configuration constants
SERVER_URL = "http://localhost:5000"  # Flask server URL
PING_MESSAGE = b"PING"
PONG_RESPONSE = b"PONG"
BAUD_RATE = 115200
TIMEOUT = 0.1  # Serial read timeout in seconds

def find_device():
    """
    Find the PPG device by checking all available serial ports.
    Send a PING message to each port and check for PONG response.
    """
    print("Searching for PPG device...")
    available_ports = list(serial.tools.list_ports.comports())
    
    if not available_ports:
        print("No serial ports found. Make sure your device is connected.")
        return None
    
    print(f"Found {len(available_ports)} serial ports. Testing each one...")
    
    for port_info in available_ports:
        port_name = port_info.device
        print(f"Trying {port_name}...")
        
        try:
            # Open serial port
            ser = serial.Serial(port=port_name, baudrate=BAUD_RATE, timeout=TIMEOUT)
            
            # Clear any pending data
            ser.reset_input_buffer()
            
            # Send PING message
            ser.write(PING_MESSAGE)
            
            # Wait for response with a timeout
            start_time = time.time()
            while (time.time() - start_time) < 2:  # 2 second timeout for response
                if ser.in_waiting > 0:
                    response = ser.read(ser.in_waiting)
                    if PONG_RESPONSE in response:
                        print(f"✓ Device found on {port_name}")
                        return port_name
                time.sleep(0.1)
                
            # No response received
            print(f"× No response from {port_name}")
            ser.close()
            
        except Exception as e:
            print(f"× Error with {port_name}: {str(e)}")
    
    print("No PPG device found on any port. Please check your connection.")
    return None

def stream_data(port_name):
    """
    Stream PPG data from the device to the server using Socket.IO.
    
    Args:
        port_name (str): The serial port name where the device is connected.
    """
    try:
        # Connect to the serial port
        ser = serial.Serial(port=port_name, baudrate=BAUD_RATE, timeout=TIMEOUT)
        
        # Create a Socket.IO client
        sio = socketio.Client()
        
        # Define Socket.IO event handlers
        @sio.event
        def connect():
            print("Connected to the server.")
        
        @sio.event
        def connect_error(data):
            print(f"Connection to server failed: {data}")
            
        @sio.event
        def disconnect():
            print("Disconnected from the server.")
        
        # Connect to the server
        print(f"Connecting to the server at {SERVER_URL}...")
        sio.connect(SERVER_URL)
        
        print("Starting data stream. Press Ctrl+C to stop.")
        
        # Main data streaming loop
        while True:
            try:
                if ser.in_waiting > 0:
                    # Read a line from the serial port
                    line = ser.readline().decode('utf-8').strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    print(f"Data: {line}")
                    
                    # Emit data to the server
                    # The server expects PPG data in the 'new_ppg_sample' event
                    sio.emit('new_ppg_sample', {'data': line})
                    
                # Small delay to prevent high CPU usage
                time.sleep(0.01)
                    
            except KeyboardInterrupt:
                print("Stopping data stream...")
                break
            except Exception as e:
                print(f"Error reading data: {str(e)}")
                continue
        
        # Clean up
        ser.close()
        sio.disconnect()
        
    except Exception as e:
        print(f"Error in stream_data: {str(e)}")
        return

if __name__ == "__main__":
    # Find the PPG device
    port_name = find_device()
    
    # If device is found, start streaming data
    if port_name:
        try:
            stream_data(port_name)
        except KeyboardInterrupt:
            print("\nBridge stopped by user.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    else:
        print("No device found. Exiting.")
        sys.exit(1) 