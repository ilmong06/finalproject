
from flask import request, jsonify
from routes.AppService.report_service import get_reports
from routes.UiRoute.secure_route import secure_bp  # secure_bp는 app.py에 등록되어 있어야 함
from routes.UiRoute.token_check import token_required

# ✅ 신고 이력 조회 라우트
@secure_bp.route("/get_reports", methods=["GET"])
@token_required
def get_reports_route():
    uuid = request.user_uuid  # JWT 토큰에서 검증된 사용자 uuid
    start = request.args.get("start_date")  # 쿼리스트링: ?start_date=2024-01-01
    end = request.args.get("end_date")      # 쿼리스트링: &end_date=2024-01-31
    keyword = request.args.get("keyword")   # 쿼리스트링: &keyword=도와줘

    reports = get_reports(uuid, start, end, keyword)
    return jsonify({"reports": reports})