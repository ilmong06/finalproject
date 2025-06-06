from flask import Blueprint, request, jsonify
from routes.Appservice import keyword_service
from routes.DataControl import validator
from Mysqldb.models import get_connection

keyword_bp = Blueprint('keyword', __name__)

@keyword_bp.route('/register_keyword', methods=['POST'])
def register_keyword_route():
    print("📡 POST /register_keyword 요청 수신됨")

    print("✅ request.content_type:", request.content_type)
    print("✅ request.form:", request.form)
    print("✅ request.values:", request.values)
    print("✅ raw data:", request.get_data())

    try:
        uuid = request.form.get("uuid") or request.values.get("uuid")
        keyword = request.form.get("keyword") or request.values.get("keyword")
        order = request.form.get("order") or request.values.get("order")

        print("📥 uuid:", uuid)
        print("📥 keyword:", keyword)
        print("📥 order:", order)

        if not uuid or not keyword or not order:
            return jsonify({"error": "Missing data"}), 400

        result = keyword_service.register_keyword(uuid, keyword, order)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
<<<<<<< Updated upstream


@keyword_bp.route('/get_keywords', methods=['POST'])
def get_keywords():
    data = request.get_json()
    uuid = data.get("uuid")
    if not uuid:
        return jsonify({"error": "UUID 누락"}), 400

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT keywd_text FROM keyword WHERE uuid = %s ORDER BY keywd_order", (uuid,))
            rows = cursor.fetchall()
            keywords = [row["keywd_text"] for row in rows]
            return jsonify({"keywords": keywords})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
=======
    

>>>>>>> Stashed changes
