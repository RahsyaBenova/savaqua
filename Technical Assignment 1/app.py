#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
import paho.mqtt.client as mqtt
import threading
import json

app = Flask(__name__)

# Contoh data dalam bentuk list
data = []

# MQTT settings
LINK_MQTT = "c0ae257fb0f1403bb96d10c278d890ee.s1.eu.hivemq.cloud"
PORT_MQTT = 8883
TOPIC_SUHU = "data/sensor/suhu"
TOPIC_KELEMBAPAN = "data/sensor/kelembapan"
USERNAME_MQTT = "savaqua"
PASSWORD_MQTT = "Savaqua123"

# MQTT client
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe([(TOPIC_SUHU, 0), (TOPIC_KELEMBAPAN, 0)])

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    print(f"Received message '{payload}' on topic '{topic}'")

    # Process message and update data
    if topic == TOPIC_SUHU:
        update_data('temperature', str(payload))
    elif topic == TOPIC_KELEMBAPAN:
        update_data('humidity', str(payload))

def update_data(key, value):
    if not data:
        new_id = 1
        new_data = {'id': new_id, 'temperature': None, 'humidity': None}
        data.append(new_data)
    
    latest_data = data[-1]
    latest_data[key] = value

    # Add a new entry if both temperature and humidity are updated
    if latest_data['temperature'] is not None and latest_data['humidity'] is not None:
        new_id = latest_data['id'] + 1
        new_data = {'id': new_id, 'temperature': None, 'humidity': None}
        data.append(new_data)

# Setup MQTT client
mqtt_client.username_pw_set(USERNAME_MQTT, PASSWORD_MQTT)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.tls_set()  # Enable TLS for secure connection

def start_mqtt():
    mqtt_client.connect(LINK_MQTT, PORT_MQTT, 60)
    mqtt_client.loop_forever()

# Start MQTT client in a separate thread
mqtt_thread = threading.Thread(target=start_mqtt)
mqtt_thread.start()

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(data)

@app.route('/api/data/<int:id>', methods=['GET'])
def get_data_by_id(id):
    result = next((item for item in data if item["id"] == id), None)
    if result:
        return jsonify(result)
    else:
        return jsonify({'message': 'Data tidak ditemukan'}), 404

@app.route('/api/data', methods=['POST'])
def add_data():
    req_data = request.get_json()
    if not req_data or 'temperature' not in req_data or 'humidity' not in req_data:
        return jsonify({'message': 'Invalid request'}), 400

    new_id = len(data) + 1
    new_data = {
        'id': new_id,
        'temperature': req_data['temperature'],
        'humidity': req_data['humidity']
    }
    data.append(new_data)
    return jsonify({'message': 'Data sukses diterima', 'data': new_data}), 201

if __name__ == '__main__':
    app.run(debug=True, port=1234)
