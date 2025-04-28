package com.example.myapplication;

import android.os.Bundle;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

public class VoiceRegisterActivity extends AppCompatActivity {

    private Button btnBack; // 뒤로가기 버튼

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_voice); // 네 XML 레이아웃 이름에 맞게 수정

        // 버튼 연결
        btnBack = findViewById(R.id.btn_back);

        // 뒤로 가기 버튼 이벤트
        btnBack.setOnClickListener(v -> {
            finish(); // 현재 액티비티 종료 -> 이전 화면으로 이동
        });
    }
}

