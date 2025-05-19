import pymysql


#mysql db endproject 스키마에 연결
def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        database='endproject',
        cursorclass=pymysql.cursors.DictCursor
    )






#중복 입력테스트
def check_user_exists(name, phnum):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) AS cnt FROM Users WHERE Name = %s AND PhNum = %s
            """, (name, phnum))
            result = cursor.fetchone()
            return result['cnt'] > 0
    finally:
        connection.close()



# 🔵 사용자 저장
def insert_user(uuid, name, phnum, reg_date):
    from Mysqldb import db
    cursor = db.cursor()
    sql = "INSERT INTO User (uuid, name, phnum, reg_date) VALUES (%s, %s, %s, %s)"
    cursor.execute(sql, (uuid, name, phnum, reg_date))
    db.commit()
    cursor.close()

# 🔵 UUID로 사용자 조회
def get_user_by_uuid(uuid):
    from Mysqldb import db
    cursor = db.cursor()
    sql = "SELECT * FROM User WHERE uuid = %s"
    cursor.execute(sql, (uuid,))
    result = cursor.fetchone()
    cursor.close()
    return result



# 🔵 전화번호로 사용자 조회
def get_user_by_phnum(phnum):
    from Mysqldb import db
    cursor = db.cursor()
    sql = "SELECT * FROM User WHERE phnum = %s"
    cursor.execute(sql, (phnum,))
    result = cursor.fetchone()
    cursor.close()
    return result

# 🔵 키워드 중복 조회
def get_keyword_by_text(uuid, keyword_text):
    from Mysqldb import db
    cursor = db.cursor()
    sql = "SELECT * FROM Keyword WHERE uuid = %s AND keyword_text = %s"
    cursor.execute(sql, (uuid, keyword_text))
    result = cursor.fetchone()
    cursor.close()
    return result

