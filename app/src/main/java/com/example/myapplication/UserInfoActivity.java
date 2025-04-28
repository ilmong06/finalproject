package com.example.myapplication;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.text.InputFilter;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
//Android 입출력
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

//CHeck log
import android.util.Log;

//Connecting UI
import android.view.View;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;

public class UserInfoActivity extends AppCompatActivity{

    // 주요 뷰 연결
    EditText etName, etPhone, etVerificationCode, etBirth, etGender, etEmergencyName, etEmergencyPhone;
    Spinner spinnerLanguage, spinnerRelation;
    Button btnRequestVerification, btnSubmit;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_userinfo);

        // EditText 연결
        etName = findViewById(R.id.plain_text_input);
        etPhone = findViewById(R.id.phone_text_input);
        etVerificationCode = findViewById(R.id.et_verification_code);
        etBirth = findViewById(R.id.birth);
        etGender = findViewById(R.id.gender);
        etEmergencyName = findViewById(R.id.et_emergency_name);
        etEmergencyPhone = findViewById(R.id.et_emergency_phone);

        // Spinner 연결
        spinnerLanguage = findViewById(R.id.spinner_language);
        spinnerRelation = findViewById(R.id.spinner_relation);

        // 버튼 연결
        btnRequestVerification = findViewById(R.id.btn_request_verification);
        btnSubmit = findViewById(R.id.btn_submit);

        // 인증요청 버튼 클릭
        btnRequestVerification.setOnClickListener(v -> {
            etVerificationCode.setEnabled(true);
            etVerificationCode.requestFocus();
            Toast.makeText(this, "인증번호를 입력하세요", Toast.LENGTH_SHORT).show();
        });

        // 인증번호 EditText 6자리 제한
        etVerificationCode.setFilters(new InputFilter[]{new InputFilter.LengthFilter(6)});

        // 완료 버튼 클릭
        btnSubmit.setOnClickListener(v -> saveUserData());
    }

    private void saveUserData() {
        // 1. 입력값 가져오기
        String name = etName.getText().toString().trim();
        String phone = etPhone.getText().toString().trim();
        String language = spinnerLanguage.getSelectedItem().toString();
        String birthRaw = etBirth.getText().toString().trim(); // 6자리 (yymmdd)
        String genderCode = etGender.getText().toString().trim(); // 1,2,3,4
        String emergencyName = etEmergencyName.getText().toString().trim();
        String emergencyPhone = etEmergencyPhone.getText().toString().trim();
        String relation = spinnerRelation.getSelectedItem().toString();

        // 2. 생년월일 변환
        int yearPrefix = Integer.parseInt(birthRaw.substring(0, 2));
        String fullBirthYear;
        if (yearPrefix <= 25) {  // 25년 이하 -> 2000년대
            fullBirthYear = "20" + birthRaw.substring(0, 2);
        } else {  // 26년 이상 -> 1900년대
            fullBirthYear = "19" + birthRaw.substring(0, 2);
        }
        String fullBirth = fullBirthYear + birthRaw.substring(2); // yyyyMMdd

        // 3. 성별 변환
        String gender = "";
        if (genderCode.equals("1") || genderCode.equals("3")) {
            gender = "남자";
        } else if (genderCode.equals("2") || genderCode.equals("4")) {
            gender = "여자";
        } else {
            gender = "기타";
        }

        // 4. DB에 저장
        saveToDatabase(name, phone, language, fullBirth, gender, emergencyName, emergencyPhone, relation);

        // 정보 저장 완료 후
        SharedPreferences prefs = getSharedPreferences("user_info", MODE_PRIVATE);
        SharedPreferences.Editor editor = prefs.edit();
        editor.putBoolean("is_registered", true);
        editor.apply();  // commit() 대신 apply() 추천

        // MainActivity로 이동
        Intent intent = new Intent(UserInfoActivity.this, MainActivity.class);
        startActivity(intent);
        finish();

    }

    private void saveToDatabase(String name, String phone, String language, String birth, String gender,
                                String emergencyName, String emergencyPhone, String relation) {
        // ✅ 여기에 SQLite 또는 Room DB 저장 로직 추가
        Log.d("UserData", "이름: " + name);
        Log.d("UserData", "전화번호: " + phone);
        Log.d("UserData", "언어: " + language);
        Log.d("UserData", "생년월일: " + birth);
        Log.d("UserData", "성별: " + gender);
        Log.d("UserData", "긴급 이름: " + emergencyName);
        Log.d("UserData", "긴급 번호: " + emergencyPhone);
        Log.d("UserData", "관계: " + relation);
    }


}
