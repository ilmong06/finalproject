import pymysql

# MySQL DB 연결 함수
def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='endproject',
        cursorclass=pymysql.cursors.DictCursor
    )

# ✅ 중복 사용자 확인 (이름 + 전화번호 기준)
def check_user_exists(name, phnum):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) AS cnt FROM userinfo WHERE Name = %s AND PhNum = %s
            """, (name, phnum))
            result = cursor.fetchone()
            return result['cnt'] > 0
    finally:
        connection.close()

# ✅ 사용자 정보 삽입 (userinfo 테이블)
def insert_user(uuid, name, phnum, birthdate, gender, reg_date):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                INSERT INTO userinfo (uuid, Name, PhNum, birthdate, gender, voicedt, reg_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (uuid, name, phnum, birthdate, gender, None, reg_date))
            connection.commit()
    finally:
        connection.close()

# ✅ 보호자 정보 삽입 (Em_noPhNum 테이블)
def insert_guardian(phnum, uuid, em_name, em_phnum, em_parent, reg_date):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                INSERT INTO Em_noPhNum (PhNum, uuid, Em_Name, Em_PhNum, Em_parent, reg_date)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (phnum, uuid, em_name, em_phnum, em_parent, reg_date))
            connection.commit()
    finally:
        connection.close()

# ✅ UUID로 사용자 조회
def get_user_by_uuid(uuid):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM userinfo WHERE uuid = %s"
            cursor.execute(sql, (uuid,))
            result = cursor.fetchone()
            return result
    finally:
        connection.close()

# ✅ 전화번호로 사용자 조회
def get_user_by_phnum(phnum):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM userinfo WHERE PhNum = %s"
            cursor.execute(sql, (phnum,))
            result = cursor.fetchone()
            return result
    finally:
        connection.close()

# ✅ GPS 정보 삽입 (ReportGPS 테이블)
def insert_gps(uuid, latitude, longitude, address):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO ReportGPS (uuid, latitude, longitude, address) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, (uuid, latitude, longitude, address))
        conn.commit()
        return True
    finally:
        conn.close()

def insert_keyword(uuid, keyword, order, add_date):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO keyword (uuid, keywd_text, keywd_order, add_date)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (uuid, keyword, order, add_date))
        conn.commit()
    finally:
        conn.close()

