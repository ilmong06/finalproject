CREATE DATABASE endproject;
use endproject;
DROP database endproject;


CREATE TABLE userinfo (
    uuid VARCHAR(36) PRIMARY KEY,         -- UUID 형식
    Name VARCHAR(8),                      -- 사용자 이름
    PhNum VARCHAR(11),                    -- 사용자 전화번호
    birthdate VARCHAR(8),                 -- 생년월일
    gender VARCHAR(6),                    -- 성별
    voicedt VARCHAR(255),                -- 음성 데이터 경로
    reg_date DATETIME                     -- 등록 날짜
);
ALTER TABLE userinfo
ADD COLUMN selected_keyword INT;


CREATE TABLE keyword (
    id INT AUTO_INCREMENT PRIMARY KEY,
    uuid VARCHAR(36),
    keywd_text VARCHAR(20),
    keywd_order INT,
    add_date DATETIME,
    FOREIGN KEY (uuid) REFERENCES userinfo(uuid)
);

CREATE TABLE Em_noPhNum (
    PhNum VARCHAR(11) PRIMARY KEY,
    uuid VARCHAR(36),
    Em_Name VARCHAR(8),
    Em_PhNum VARCHAR(11),
    Em_parent VARCHAR(8),
    reg_date DATETIME,
    FOREIGN KEY (uuid) REFERENCES userinfo(uuid)
);
CREATE TABLE ReportGPS (
    id INT AUTO_INCREMENT PRIMARY KEY,
    uuid VARCHAR(36),
    latitude DOUBLE,
    longitude DOUBLE,
    report_time DATETIME,
    FOREIGN KEY (uuid) REFERENCES userinfo(uuid)
);
CREATE TABLE voice (
    id INT AUTO_INCREMENT PRIMARY KEY,
    keyword_id INT,
    uuid VARCHAR(36),
    voice_index INT,
    voice_path VARCHAR(255),
    reg_date DATETIME,
    FOREIGN KEY (keyword_id) REFERENCES keyword(id),
    FOREIGN KEY (uuid) REFERENCES userinfo(uuid)
);
ALTER TABLE userinfo
ADD CONSTRAINT fk_selected_keyword
FOREIGN KEY (selected_keyword) REFERENCES keyword(id);

select * from userinfo where Name="최동욱";
select * from em_nophnum where Em_Name="최재민";

select * from userinfo;
select * from keyword;
select * from em_nophnum;
select * from ReportGPS;
select * from voice;

DROP TABLE IF EXISTS userinfo;
DROP TABLE IF EXISTS ReportGPS;
DROP TABLE IF EXISTS Em_noPhNum;
DROP TABLE IF EXISTS keyword;


INSERT INTO userinfo (
    uuid, Name, PhNum, birthdate, gender, voicedt, reg_date
) VALUES (
    UUID(),
    '이순신',
    '01098765432',
    '01010101',
    'male',
    'uploads/voicedata/example.wav',
    NOW()
);

SELECT
    u.uuid,
    u.Name AS 사용자이름,
    u.PhNum AS 사용자번호,

    e.Em_Name AS 보호자이름,
    e.Em_PhNum AS 보호자번호,
    e.Em_parent AS 보호자관계,
    e.reg_date AS 보호자등록일,

    k.keywd_text AS 키워드,
    k.keywd_order AS 키워드순서,
    k.add_date AS 키워드등록일
FROM userinfo u
LEFT JOIN Em_noPhNum e ON u.uuid = e.uuid
LEFT JOIN keyword k ON u.uuid = k.uuid
WHERE u.Name = 'tt';
 #사용자와 보호자 정보를 함께 조회하는 SQL


INSERT INTO Em_noPhNum (
    PhNum,
    uuid,
    Em_Name,
    Em_PhNum,
    Em_parent,
    reg_date
) VALUES (
    '01012345678',  -- 사용자 번호
    'ab3ab66a-41cd-11f0-9a15-d8f8837c2602',  -- userinfo 테이블에 존재하는 uuid
    '최재민',       -- 보호자 성명
    '01098765432',  -- 보호자 연락처
    '부',           -- 보호자 관계
    NOW()           -- 현재 날짜/시간
);#  보호자 정보 임의 등록 예시

