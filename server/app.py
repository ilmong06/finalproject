from flask import Flask,render_template_string
from flask_cors import CORS


from routes.UiRoute.user_info_route import user_info_bp #userinforoute
from routes.UiRoute.secure_route import secure_bp
from routes.UiRoute.register_route import register_bp
from routes.UiRoute.keyword_route import keyword_bp
# ✅ Android layout 디렉토리 경로 지정
ANDROID_LAYOUT_PATH = "C:/Users/worker/Desktop/python/finaltest/worr/app/src/main/res/layout"


app = Flask(__name__)
CORS(app)

# Blueprint 등록
app.register_blueprint(user_info_bp, url_prefix='/api')
app.register_blueprint(secure_bp, url_prefix="/api")
app.register_blueprint(keyword_bp, url_prefix="/api")
app.register_blueprint(register_bp, url_prefix="/api")

#xml 파일 못읽었을시 예외처리 <- 필수는 아님
def render_xml_file(filename):
    try:
        full_path = f"{ANDROID_LAYOUT_PATH}/{filename}"
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        # HTML에서 < >가 깨지지 않도록 escape 처리
        return render_template_string("<pre>{{ content }}</pre>", content=content)
    except FileNotFoundError:
        return f"파일 {filename}을 찾을 수 없습니다.", 404

#1. activity_main.xml 앱 메인 화면 레이아웃
@app.route('/')
def mainpage():
    return render_xml_file("activity_main.xml")

#2.activity_keyword.xml 사용자 정의 키워드 등록 및 녹음 화면
@app.route('/keyword')
def insertkeyword():
    return render_xml_file("activity_keyword.xml")

#3.activity_userinfo.xml 사용자 정보 입력 화면
@app.route('/userinfo')
def inserttable():
    return render_xml_file("activity_userinfo.xml")

#4.activity_voice.xml 사용자 음성 등록 화면
@app.route('/voice')
def insertvoice():
    return render_xml_file("activity_voice.xml")

#5. activity_admin.xml 관리자 전용 기능 화면
@app.route('/admin')
def adminpage():
    return render_xml_file("activity_admin.xml")


if __name__ == '__main__':
    
    print("\n[현재 등록된 라우트 목록]")
    for rule in app.url_map.iter_rules():
        print(f"{rule.methods} -> {rule}")
    
    app.run(host='0.0.0.0', port=5001, debug=True)


