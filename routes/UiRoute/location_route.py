from flask import Blueprint, request, jsonify
from Mysqldb import models
from datetime import datetime

location_bp = Blueprint('location', __name__)

# 🔵 긴급 위치 저장
@location_bp.route('/location', methods=['POST'])
def location():
    try:
        uuid = request.form['uuid']
        latitude = request.form['latitude']
        longitude = request.form['longitude']

        if not uuid or not latitude or not longitude:
            return jsonify({"error": "Missing data"}), 400

        # DB에 위치 저장
        models.save_location(uuid, latitude, longitude, datetime.now())

        print(f"🚨 긴급 위치 수신: UUID={uuid}, 위도={latitude}, 경도={longitude}")

        return jsonify({"message": "Location saved successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
