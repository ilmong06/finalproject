from Mysqldb.models import get_connection
from Mysqldb import models
import pymysql

# ✅ 신고 이력 조회 서비스
def get_reports(uuid, start_date=None, end_date=None, keyword=None):
    connection = get_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    # 기본 쿼리
    query = '''
        SELECT report_time, latitude, longitude, address, keyword
        FROM ReportGPS
        WHERE uuid = %s
    '''
    params = [uuid]

    # 날짜 필터
    if start_date:
        query += " AND report_time >= %s"
        params.append(start_date)
    if end_date:
        query += " AND report_time <= %s"
        params.append(end_date)

    # 키워드 필터 (LIKE 검색)
    if keyword:
        query += " AND keyword LIKE %s"
        params.append(f"%{keyword}%")

    try:
        cursor.execute(query, params)
        results = cursor.fetchall()
        return results
    except Exception as e:
        print("❌ 신고 이력 조회 실패:", e)
        return []
    finally:
        cursor.close()
        connection.close()

def save_location(uuid, lat, lon, address):
    return models.insert_gps(uuid, lat, lon, address)