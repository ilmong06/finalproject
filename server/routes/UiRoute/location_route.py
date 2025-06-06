import requests
from flask import Blueprint, request, jsonify
from Mysqldb import models
from datetime import datetime
from routes.Appservice import report_service


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


#kakao api를 이용한 상세주소변환
def get_address_from_kakao(lat, lng):
    try:
        KAKAO_REST_API_KEY = "b635cf490ef3c48fb2f1ce467c4a51aa"  # 카카오 api키
        headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
        url = f"https://dapi.kakao.com/v2/local/geo/coord2address.json?x={lng}&y={lat}"

        response = requests.get(url, headers=headers)
        result = response.json()

        if "documents" in result and len(result["documents"]) > 0:
            return result["documents"][0]["address"]["address_name"]
        else:
            return "주소 정보 없음"
    except Exception as e:
        return f"주소 변환 오류: {str(e)}"
    
@location_bp.route('/send_gps', methods=['POST'])
def send_gps():
    try:
        data = request.get_json()
        uuid = data.get('uuid')
        lat = data.get('latitude')
        lng = data.get('longitude')

        if not uuid or lat is None or lng is None:
            return jsonify({"error": "필수값 누락"}), 400

        address = get_address_from_kakao(lat, lng)  # ✅ 주소 변환

        # DB 저장
        result = report_service.save_location(uuid, lat, lng, address)

        return jsonify({"message": "위치 저장 성공", "address": address}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

