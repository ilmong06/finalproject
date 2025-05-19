use endproject;

create table userinfo ( #사용자 테이블
		uuid varchar(36) primary key, #주요키는 uuid 자동생성 36자까지로 되어있음
    	Name varchar(8), #사용자이름
    	PhNum varchar(11), # 사용자연락처  ex 01023899074
		birthdate varchar(8), # 사용자주민등록번호 앞자리 6자리 뒷자리 하나 1) 6자리 입력 2) 앞 두자리로 연도 확인 3) 20 or 19 +6자리 4) 8자리 반환
		gender varchar(6),  #사용자성별 male, female 둘중하나 
		voicedt varchar(255), #음성데이터 파일[voicedata]로uploads 폴더에 저장해 디렉토리로 불러오기 
		reg_date datetime #사용자 등록날짜
);


CREATE TABLE keyword ( #키워드 테이블
    	id INT AUTO_INCREMENT PRIMARY KEY, #주요키는 입력된 순서로 표기
    	uuid VARCHAR(36), #userinfo에서 가져온 uuid그대로 적용
    	keywd_text VARCHAR(20), #키워드 
		keywd_order INT, #키워드들를 번호로 구분해 따로 저장
    	add_date DATETIME, #키워드 추가한 날짜
    	FOREIGN KEY (uuid) REFERENCES userinfo(uuid) #외래키로 userinfo(uuid) 참조해 그대로 사용
);


create table Em_noPhNum( #보호자정보 테이블
	PhNum varchar(11) primary key, #주요키는 사용자연락처 
	uuid VARCHAR(36), #userinfo에서 가져온 uuid그대로 적용
	Em_Name varchar(8), #보호자성명
	Em_PhNum varchar(11), #보호자 연락처
	Em_parent varchar(8), #보호자 관계 ex 부,모, 할아버지 등등 혹시몰라 8자까지

	reg_date DATETIME, #보호자 정보 추가날짜
	FOREIGN KEY (uuid) REFERENCES userinfo(uuid) #외래키로 userinfo(uuid) 참조해 그대로 사용
);

CREATE TABLE ReportGPS (
   	 id INT AUTO_INCREMENT PRIMARY KEY, #순번대로 정리해 기록을 최신순으로 불러오기
   	 uuid VARCHAR(36), #해당기능을 사용한 사용자의 고유uid
   	 latitude DOUBLE, #위도
    	 longitude DOUBLE, #경도
   	 report_time DATETIME, #신고한 날짜
   	 FOREIGN KEY (uuid) REFERENCES userinfo(uuid) #외래키로 userinfo(uuid) 참조해 그대로 사용
);

select * from userinfo where Name="최동욱";
select * from em_nophnum where Em_Name="최재민";

select * from userinfo;
select * from keyword;
select * from em_nophnum;
select * from ReportGPS;

